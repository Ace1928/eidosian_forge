from __future__ import annotations
import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from . import __version__
from .config import PeftConfig
from .tuners import (
from .utils import (
class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForSequenceClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "SEQ_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str='default') -> None:
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {'classifier', 'score'}
        else:
            self.modules_to_save.update({'classifier', 'score'})
        for name, _ in self.base_model.named_children():
            if any((module_name in name for module_name in self.modules_to_save)):
                self.cls_layer_name = name
                break
        _set_trainable(self, adapter_name)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, task_ids=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs['task_ids'] = task_ids
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get('position_ids', None) is not None:
            warnings.warn('Position ids are not supported for parameter efficient tuning. Ignoring position ids.')
            kwargs['position_ids'] = None
        kwargs.update({'attention_mask': attention_mask, 'labels': labels, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states, 'return_dict': return_dict})
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get('token_type_ids', None) is not None:
                kwargs['token_type_ids'] = torch.cat((torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device), kwargs['token_type_ids']), dim=1).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update({'input_ids': input_ids, 'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states, 'return_dict': return_dict, 'past_key_values': past_key_values})
        if 'past_key_values' in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if 'past_key_values' not in fwd_params:
                raise ValueError('Model does not support past key values which are required for prefix tuning.')
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if 'dropout' in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)
            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = 'regression'
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = 'single_label_classification'
                    else:
                        self.config.problem_type = 'multi_label_classification'
                if self.config.problem_type == 'regression':
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == 'single_label_classification':
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == 'multi_label_classification':
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return (loss,) + output if loss is not None else output
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)