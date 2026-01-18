import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
class MegatronEngine(torch.nn.Module):
    """
    Megatron-LM model wrapper

    Args:
        accelerator (:class:`~accelerate.Accelerator`): The accelerator object to use.
        model: Megatron-LM model
        optimizer: Megatron-LM optimizer
        lr_scheduler: Megatron-LM lr scheduler
    """

    def __init__(self, accelerator, model, optimizer, scheduler):
        super().__init__()
        self.module = model
        self.base_model = model[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        args = get_args()
        if accelerator.state.megatron_lm_plugin.custom_train_step_class is not None:
            self.train_step_handler = accelerator.state.megatron_lm_plugin.custom_train_step_class(args, **accelerator.state.megatron_lm_plugin.custom_train_step_kwargs)
        elif args.model_type_name == 'bert':
            self.train_step_handler = BertTrainStep(args)
        elif args.model_type_name == 'gpt':
            self.train_step_handler = GPTTrainStep(args)
        elif args.model_type_name == 't5':
            self.train_step_handler = T5TrainStep(args)
        else:
            raise ValueError(f'Unsupported model type: {args.model_type_name}')
        self.optimizer.skipped_iter = False
        self.total_loss_dict = {}
        self.eval_total_loss_dict = {}
        self.iteration = 0
        self.report_memory_flag = True
        if args.tensorboard_dir is not None:
            write_args_to_tensorboard()

    def train(self):
        for model_module in self.module:
            model_module.train()
        self.log_eval_results()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def train_step(self, **batch_data):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """
        args = get_args()
        timers = get_timers()
        if len(batch_data) > 0:
            data_chunks = []
            if args.num_micro_batches > 1:
                for i in range(0, args.num_micro_batches):
                    data_chunks.append({k: v[i * args.micro_batch_size:(i + 1) * args.micro_batch_size] for k, v in batch_data.items()})
            else:
                data_chunks = [batch_data]
        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))] if len(batch_data) > 0 else [None] * len(self.module)
        else:
            batch_data_iterator = iter(data_chunks) if len(batch_data) > 0 else None
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in self.module:
                partition.zero_grad_buffer()
        self.optimizer.zero_grad()
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(self.train_step_handler.forward_step, batch_data_iterator, self.module, self.optimizer, None, forward_only=False)
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
        timers('backward-reduce-model-grads').start()
        self.optimizer.reduce_model_grads(args, timers)
        timers('backward-reduce-model-grads').stop()
        timers('optimizer').start()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(args, timers)
        timers('optimizer').stop()
        if update_successful:
            timers('backward-gather-model-params').start()
            self.optimizer.gather_model_params(args, timers)
            timers('backward-gather-model-params').stop()
        if update_successful:
            if self.scheduler is not None:
                increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                self.scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1
        self.optimizer.skipped_iter = not update_successful
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return (loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad)
        return ({}, skipped_iter, grad_norm, num_zeros_in_grad)

    def eval_step(self, **batch_data):
        """
        Evaluation step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to evaluate on.
        """
        args = get_args()
        data_chunks = []
        if args.num_micro_batches > 1:
            for i in range(0, args.num_micro_batches):
                data_chunks.append({k: v[i * args.micro_batch_size:(i + 1) * args.micro_batch_size] for k, v in batch_data.items()})
        else:
            data_chunks = [batch_data]
        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))]
        else:
            batch_data_iterator = iter(data_chunks)
        forward_backward_func = get_forward_backward_func()
        loss_dicts = forward_backward_func(self.train_step_handler.forward_step, batch_data_iterator, self.module, optimizer=None, timers=None, forward_only=True)
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
        args.consumed_valid_samples += mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced
        else:
            return {}

    def forward(self, **batch_data):
        args = get_args()
        if self.module[0].training:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(**batch_data)
            self.iteration += 1
            if args.tensorboard_dir is not None:
                loss_scale = self.optimizer.get_loss_scale().item()
                params_norm = None
                if args.log_params_norm:
                    params_norm = calc_params_l2_norm(self.model)
                self.report_memory_flag = training_log(loss_dict, self.total_loss_dict, self.optimizer.param_groups[0]['lr'], self.iteration, loss_scale, self.report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad)
        else:
            loss_dict = self.eval_step(**batch_data)
            if args.tensorboard_dir is not None:
                for key in loss_dict:
                    self.eval_total_loss_dict[key] = self.eval_total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
                    self.eval_total_loss_dict[key + '_num_iters'] = self.eval_total_loss_dict.get(key + '_num_iters', torch.cuda.FloatTensor([0.0])) + torch.cuda.FloatTensor([1.0])
        loss = torch.tensor(0.0, device=args.local_rank)
        for key in loss_dict:
            if len(loss_dict[key].shape) == 0:
                loss += loss_dict[key]
        logits = None
        if 'logits' in loss_dict:
            logits = loss_dict['logits']
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss, logits=logits)
        return loss

    def log_eval_results(self):
        args = get_args()
        if args.tensorboard_dir is None or self.iteration == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f'validation loss at iteration {self.iteration} | '
        for key in self.eval_total_loss_dict:
            if key.endswith('_num_iters'):
                continue
            value = self.eval_total_loss_dict[key] / self.eval_total_loss_dict[key + '_num_iters']
            string += f'{key} value: {value} | '
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f'{key} PPL: {ppl} | '
            if writer:
                writer.add_scalar(f'{key} validation', value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f'{key} validation ppl', ppl, self.iteration)
        length = len(string) + 1
        print_rank_last('-' * length)
        print_rank_last(string)
        print_rank_last('-' * length)
        self.eval_total_loss_dict = {}

    def save_checkpoint(self, output_dir):
        self.log_eval_results()
        args = get_args()
        args.save = output_dir
        torch.distributed.barrier()
        save_checkpoint(self.iteration, self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()

    def load_checkpoint(self, input_dir):
        args = get_args()
        args.load = input_dir
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        torch.distributed.barrier()
        iteration = load_checkpoint(self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()
        self.iteration = iteration
        if args.fp16 and self.iteration == 0:
            self.optimizer.reload_model_params()

    def megatron_generate(self, inputs, attention_mask=None, max_length=None, max_new_tokens=None, num_beams=None, temperature=None, top_k=None, top_p=None, length_penalty=None, **kwargs):
        """
        Generate method for GPT2 model. This method is used for inference. Supports both greedy and beam search along
        with sampling. Refer the Megatron-LM repo for more details

        Args:
            inputs (torch.Tensor): input ids
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            max_length (int, optional): max length of the generated sequence. Defaults to None.
            Either this or max_new_tokens should be provided.
            max_new_tokens (int, optional): max number of tokens to be generated. Defaults to None.
            Either this or max_length should be provided.
            num_beams (int, optional): number of beams to use for beam search. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top k tokens to consider for sampling. Defaults to 0.0.
            top_p (float, optional): tokens in top p probability are considered for sampling. Defaults to 0.0.
            length_penalty (float, optional): length penalty for beam search. Defaults to None.
            kwargs: additional key-value arguments
        """
        args = get_args()
        if args.model_type_name != 'gpt':
            raise NotImplementedError('Generate method is not implemented for this model')
        if args.data_parallel_size > 1:
            raise ValueError('Generate method requires data parallelism to be 1')
        if args.sequence_parallel:
            raise ValueError('Generate method requires sequence parallelism to be False')
        if args.recompute_granularity is not None:
            raise ValueError('Checkpoint activations cannot be set for inference')
        if args.vocab_file is None:
            raise ValueError('Vocab file is required for inference')
        if max_length is None and max_new_tokens is None:
            raise ValueError('`max_length` or `max_new_tokens` are required for inference')
        if temperature is None:
            temperature = 1.0
        elif not 0.0 < temperature <= 100.0:
            raise ValueError('temperature must be a positive number less than or equal to 100.0')
        if top_k is None:
            top_k = 0
        elif not 0 <= top_k <= 1000:
            raise ValueError('top_k must be a positive number less than or equal to 1000')
        if top_p is None:
            top_p = 0.0
        elif top_p > 0.0 and top_k > 0.0:
            raise ValueError('top_p and top_k sampling cannot be set together')
        elif not 0.0 <= top_p <= 1.0:
            raise ValueError('top_p must be less than or equal to 1.0')
        top_p_decay = kwargs.get('top_p_decay', 0.0)
        if not 0.0 <= top_p_decay <= 1.0:
            raise ValueError('top_p_decay must be less than or equal to 1.0')
        top_p_bound = kwargs.get('top_p_bound', 0.0)
        if not 0.0 <= top_p_bound <= 1.0:
            raise ValueError('top_p_bound must be less than or equal to 1.0')
        add_BOS = kwargs.get('add_BOS', False)
        if not isinstance(add_BOS, bool):
            raise ValueError('add_BOS must be a boolean')
        beam_width = num_beams
        if beam_width is not None:
            if not isinstance(beam_width, int):
                raise ValueError('beam_width must be an integer')
            if beam_width < 1:
                raise ValueError('beam_width must be greater than 0')
            if inputs.shape[0] > 1:
                return 'When doing beam_search, batch size must be 1'
        tokenizer = get_tokenizer()
        stop_token = kwargs.get('stop_token', tokenizer.eod)
        if stop_token is not None:
            if not isinstance(stop_token, int):
                raise ValueError('stop_token must be an integer')
        if length_penalty is None:
            length_penalty = 1.0
        sizes_list = None
        prompts_tokens_tensor = None
        prompts_length_tensor = None
        if torch.distributed.get_rank() == 0:
            if attention_mask is None:
                prompts_length_tensor = torch.cuda.LongTensor([inputs.shape[1]] * inputs.shape[0])
            else:
                prompts_length_tensor = attention_mask.sum(axis=-1).cuda()
            if max_new_tokens is None:
                max_new_tokens = max_length - inputs.shape[1]
            if max_new_tokens <= 0:
                raise ValueError('max_new_tokens must be greater than 0')
            if add_BOS:
                max_length = max_new_tokens + inputs.shape[1] + 1
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - (inputs.shape[1] + 1)
                padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([torch.unsqueeze(padding[:, 0], axis=-1), inputs.cuda(), padding], axis=-1)
            else:
                max_length = max_new_tokens + inputs.shape[1]
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - inputs.shape[1]
                padding = torch.cuda.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([inputs.cuda(), padding], axis=-1)
            sizes_list = [prompts_tokens_tensor.size(0), prompts_tokens_tensor.size(1)]
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)
        sizes = sizes_tensor.tolist()
        context_tokens_tensor = broadcast_tensor(sizes, torch.int64, tensor=prompts_tokens_tensor, rank=0)
        context_length_tensor = broadcast_tensor(sizes[0], torch.int64, tensor=prompts_length_tensor, rank=0)
        random_seed = kwargs.get('random_seed', 0)
        torch.random.manual_seed(random_seed)
        unwrapped_model = unwrap_model(self.base_model, (torchDDP, LocalDDP, Float16Module))
        if beam_width is not None:
            tokens, _ = beam_search_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, beam_width, stop_token=stop_token, num_return_gen=1, length_penalty=length_penalty)
        else:
            tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, return_output_log_probs=False, top_k=top_k, top_p=top_p, top_p_decay=top_p_decay, top_p_bound=top_p_bound, temperature=temperature, use_eod_token_for_early_termination=True)
        return tokens