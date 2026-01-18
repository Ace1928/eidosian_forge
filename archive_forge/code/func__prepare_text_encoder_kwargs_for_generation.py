import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
def _prepare_text_encoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str]=None, guidance_scale: Optional[float]=None) -> Dict[str, Any]:
    encoder = self.get_text_encoder()
    if hasattr(encoder, '_hf_hook'):
        encoder._hf_hook.io_same_device = True
    irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
    if not encoder_accepts_wildcard:
        encoder_kwargs = {argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature}
    model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
    encoder_kwargs['return_dict'] = True
    encoder_kwargs[model_input_name] = inputs_tensor
    last_hidden_state = encoder(**encoder_kwargs).last_hidden_state
    if guidance_scale is not None and guidance_scale > 1:
        last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
        if 'attention_mask' in model_kwargs:
            model_kwargs['attention_mask'] = torch.concatenate([model_kwargs['attention_mask'], torch.zeros_like(model_kwargs['attention_mask'])], dim=0)
    model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=last_hidden_state)
    return model_kwargs