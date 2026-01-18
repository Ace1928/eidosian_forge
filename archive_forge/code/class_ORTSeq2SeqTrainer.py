from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput
from transformers.utils import is_accelerate_available, logging
from .trainer import ORTTrainer
class ORTSeq2SeqTrainer(ORTTrainer):

    def evaluate(self, eval_dataset: Optional[Dataset]=None, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='eval', **gen_kwargs) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None and (self.args.generation_max_length is not None):
            gen_kwargs['max_length'] = self.args.generation_max_length
        if gen_kwargs.get('num_beams') is None and self.args.generation_num_beams is not None:
            gen_kwargs['num_beams'] = self.args.generation_num_beams
        self._gen_kwargs = gen_kwargs
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='test', **gen_kwargs) -> 'PredictionOutput':
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None and (self.args.generation_max_length is not None):
            gen_kwargs['max_length'] = self.args.generation_max_length
        if gen_kwargs.get('num_beams') is None and self.args.generation_num_beams is not None:
            gen_kwargs['num_beams'] = self.args.generation_num_beams
        self._gen_kwargs = gen_kwargs
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]]=None, **gen_kwargs) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)
        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
        if 'num_beams' in gen_kwargs and gen_kwargs['num_beams'] is None:
            gen_kwargs.pop('num_beams')
        if 'max_length' in gen_kwargs and gen_kwargs['max_length'] is None:
            gen_kwargs.pop('max_length')
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus
        generation_inputs = inputs.copy()
        if 'labels' in generation_inputs and 'decoder_input_ids' in generation_inputs and (generation_inputs['labels'].shape == generation_inputs['decoder_input_ids'].shape):
            generation_inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False
        gen_config = self.model.generation_config
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None
        if self.args.prediction_loss_only:
            return (loss, None, None)
        if has_labels:
            labels = inputs['labels']
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None
        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        elif self.model.config.pad_token_id is not None:
            pad_token_id = self.model.config.pad_token_id
        else:
            raise ValueError('Pad_token_id must be set in the configuration of the model, in order to pad tensors')
        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, :tensor.shape[-1]] = tensor
        return padded_tensor