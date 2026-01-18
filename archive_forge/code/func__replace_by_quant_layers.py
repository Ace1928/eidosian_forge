import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod
from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen
def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str=''):
    """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
    QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=self.desc_act, group_size=self.group_size, bits=self.bits, disable_exllama=self.disable_exllama or self.exllama_version != ExllamaVersion.ONE, disable_exllamav2=self.disable_exllama or self.exllama_version != ExllamaVersion.TWO)
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        layer = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            device = get_device(layer)
            delattr(module, attr)
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
            elif isinstance(layer, nn.Conv2d):
                in_features = layer.in_channels
                out_features = layer.out_channels
            elif isinstance(layer, Conv1D):
                in_features = layer.weight.shape[0]
                out_features = layer.weight.shape[1]
            if not self.desc_act or self.group_size == -1:
                new_layer = QuantLinear(self.bits, self.group_size, in_features, out_features, True, use_cuda_fp16=self.use_cuda_fp16, weight_dtype=layer.weight.dtype)
            else:
                new_layer = QuantLinear(self.bits, self.group_size, in_features, out_features, True, weight_dtype=layer.weight.dtype)
            new_layer.device = device
            setattr(module, attr, new_layer.to(device))
    for name1, child in module.named_children():
        self._replace_by_quant_layers(child, names, name + '.' + name1 if name != '' else name1)