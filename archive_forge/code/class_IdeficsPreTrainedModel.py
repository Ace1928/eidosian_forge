from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
@add_start_docstrings('The bare LLaMA Model outputting raw hidden-states without any specific head on top.', LLAMA_START_DOCSTRING)
class IdeficsPreTrainedModel(PreTrainedModel):
    config_class = IdeficsConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['IdeficsDecoderLayer', 'IdeficsGatedCrossAttentionLayer']
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool=False) -> PretrainedConfig:
        _is_bettertransformer = getattr(cls, 'use_bettertransformer', False)
        if _is_bettertransformer:
            return config
        if not hard_check_only:
            config._attn_implementation = 'sdpa'
        return config