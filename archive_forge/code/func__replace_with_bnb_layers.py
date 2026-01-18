import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from accelerate.utils.imports import (
from ..big_modeling import dispatch_model, init_empty_weights
from .dataclasses import BnbQuantizationConfig
from .modeling import (
def _replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=None, current_key_name=None):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    import bitsandbytes as bnb
    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            current_key_name_str = '.'.join(current_key_name)
            proceed = True
            for key in modules_to_not_convert:
                if key in current_key_name_str and key + '.' in current_key_name_str or key == current_key_name_str:
                    proceed = False
                    break
            if proceed:
                if bnb_quantization_config.load_in_8bit:
                    bnb_module = bnb.nn.Linear8bitLt(module.in_features, module.out_features, module.bias is not None, has_fp16_weights=False, threshold=bnb_quantization_config.llm_int8_threshold)
                elif bnb_quantization_config.load_in_4bit:
                    bnb_module = bnb.nn.Linear4bit(module.in_features, module.out_features, module.bias is not None, bnb_quantization_config.bnb_4bit_compute_dtype, compress_statistics=bnb_quantization_config.bnb_4bit_use_double_quant, quant_type=bnb_quantization_config.bnb_4bit_quant_type)
                else:
                    raise ValueError("load_in_8bit and load_in_4bit can't be both False")
                bnb_module.weight.data = module.weight.data
                if module.bias is not None:
                    bnb_module.bias.data = module.bias.data
                bnb_module.requires_grad_(False)
                setattr(model, name, bnb_module)
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = _replace_with_bnb_layers(module, bnb_quantization_config, modules_to_not_convert, current_key_name)
            has_been_replaced = has_been_replaced | _has_been_replaced
        current_key_name.pop(-1)
    return (model, has_been_replaced)