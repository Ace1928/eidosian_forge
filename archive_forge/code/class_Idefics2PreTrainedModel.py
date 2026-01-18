import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...utils import (
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
@add_start_docstrings('The bare Idefics2 Model outputting raw hidden-states without any specific head on top.', IDEFICS2_START_DOCSTRING)
class Idefics2PreTrainedModel(PreTrainedModel):
    config_class = Idefics2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Idefics2VisionAttention', 'Idefics2MLP', 'Idefics2PerceiverLayer', 'Idefics2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range if hasattr(self.config, 'initializer_range') else self.config.text_config.initializer_range
        if hasattr(module, 'class_embedding'):
            module.class_embedding.data.normal_(mean=0.0, std=std)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _autoset_attn_implementation(cls, config, use_flash_attention_2: bool=False, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None, check_device_map: bool=True, **kwargs):
        """
        Overrides the method in `PreTrainedModel` to update the vision config with the correct attention implementation
        """
        config = super()._autoset_attn_implementation(config=config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map, check_device_map=check_device_map, **kwargs)
        config.vision_config._attn_implementation = config._attn_implementation
        return config