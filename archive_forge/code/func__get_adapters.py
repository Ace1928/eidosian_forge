import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
def _get_adapters(self):
    if self.config.adapter_attn_dim is None:
        raise ValueError(f'{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.')
    adapter_weights = {}
    for name, module in self.named_modules():
        if isinstance(module, Wav2Vec2AttnAdapterLayer):
            for param_name, param in module.named_parameters():
                adapter_weights['.'.join([name, param_name])] = param
    if isinstance(self, Wav2Vec2ForCTC):
        for name, param in self.lm_head.named_parameters():
            adapter_weights['.'.join(['lm_head', name])] = param
    return adapter_weights