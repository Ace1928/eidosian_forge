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
class T5TrainStep(AbstractTrainStep):
    """
    T5 train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__('T5TrainStep')
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    @staticmethod
    def attn_mask_postprocess(attention_mask):
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    @staticmethod
    def get_decoder_mask(seq_length, device):
        attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
        attention_mask = attention_mask < 0.5
        return attention_mask

    @staticmethod
    def get_enc_dec_mask(attention_mask, dec_seq_length, device):
        batch_size, _ = attention_mask.shape
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = torch.ones((batch_size, dec_seq_length, 1), device=device)
        attention_mask_bss = attention_mask_bs1 * attention_mask_b1s
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    def get_batch_func(self, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Build the batch."""
            keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)
            tokens_enc = data_b['text_enc'].long()
            tokens_dec = data_b['text_dec'].long()
            labels = data_b['labels'].long()
            loss_mask = data_b['loss_mask'].float()
            enc_mask = data_b['enc_mask'] < 0.5
            dec_mask = data_b['dec_mask'] < 0.5
            enc_dec_mask = data_b['enc_dec_mask'] < 0.5
            return (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask)

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())
            tokens_enc = data['input_ids'].long()
            labels = data['labels'].long()
            loss_mask = (labels != -100).to(torch.float)
            if 'decoder_input_ids' in data:
                tokens_dec = data['decoder_input_ids'].long()
            else:
                tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
                tokens_dec[..., 1:] = labels[..., :-1].clone()
                tokens_dec[..., 0] = 0
                tokens_dec.masked_fill_(tokens_dec == -100, 0)
            enc_mask = T5TrainStep.attn_mask_postprocess(data['attention_mask'].long())
            dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
            enc_dec_mask = T5TrainStep.get_enc_dec_mask(data['attention_mask'].long(), tokens_dec.shape[1], tokens_dec.device)
            return (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask)
        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):

        def loss_func(loss_mask, output_tensor):
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])
            return (loss, {'lm loss': averaged_losses[0]})
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch(data_iterator)
            output_tensor = model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=lm_labels)
            return (output_tensor, partial(self.loss_func, loss_mask))
        return forward_step