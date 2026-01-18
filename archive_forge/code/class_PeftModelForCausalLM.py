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
class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str='default') -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, task_ids=None, **kwargs):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == 'mpt':
                if inputs_embeds is not None:
                    raise AssertionError('forward in MPTForCausalLM does not support inputs_embeds')
                return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
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
        if kwargs.get('token_type_ids', None) is not None:
            warnings.warn('Token type ids are not supported for parameter efficient tuning. Ignoring token type ids')
            kwargs['token_type_ids'] = None
        kwargs.update({'attention_mask': attention_mask, 'labels': labels, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states, 'return_dict': return_dict})
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs['labels'] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, *args, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, 'model'):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(*args, **kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor]=None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse('4.38.0')
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse('4.36.0')
        transformers_new_cache_archs = ['llama', 'mistral', 'persimmon', 'phi']
        uses_cache = uses_transformers_4_38 or (uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs['task_ids'] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and model_kwargs['past_key_values'] is not None:
                if model_kwargs['past_key_values'][0][0].shape[-2] >= model_kwargs['input_ids'].shape[1]:
                    model_kwargs['input_ids'] = model_kwargs['input_ids'][:, -1:]
            if model_kwargs.get('attention_mask', None) is not None:
                size = (model_kwargs['input_ids'].shape[0], peft_config.num_virtual_tokens)
                prefix_attention_mask = torch.ones(size).to(model_kwargs['input_ids'].device)
                model_kwargs['attention_mask'] = torch.cat((prefix_attention_mask, model_kwargs['attention_mask']), dim=1)
            if model_kwargs.get('position_ids', None) is not None:
                warnings.warn('Position ids are not supported for parameter efficient tuning. Ignoring position ids.')
                model_kwargs['position_ids'] = None
            if kwargs.get('token_type_ids', None) is not None:
                warnings.warn('Token type ids are not supported for parameter efficient tuning. Ignoring token type ids')
                kwargs['token_type_ids'] = None
            if model_kwargs['past_key_values'] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs['input_ids'].shape[0])
                model_kwargs['past_key_values'] = past_key_values
            elif model_kwargs['past_key_values'] is None:
                inputs_embeds = self.word_embeddings(model_kwargs['input_ids'])
                prompts = self.get_prompt(batch_size=model_kwargs['input_ids'].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs['inputs_embeds'] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs['input_ids'] = None
        _ = model_kwargs.pop('cache_position', None)
        return model_kwargs