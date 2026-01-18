from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...image_processing_utils import select_best_resolution
from ...modeling_outputs import ModelOutput
from ...utils import (
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava_next import LlavaNextConfig
@add_start_docstrings('The bare LLaMA Model outputting raw hidden-states without any specific head on top.', LLAVA_NEXT_START_DOCSTRING)
class LlavaNextPreTrainedModel(PreTrainedModel):
    config_class = LlavaNextConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['LlavaNextVisionAttention']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else self.config.text_config.initializer_range
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

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa