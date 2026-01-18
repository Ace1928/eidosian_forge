import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
def _load_state_dict_into_meta_model(model, state_dict, loaded_state_dict_keys, start_prefix, expected_keys, device_map=None, offload_folder=None, offload_index=None, state_dict_folder=None, state_dict_index=None, dtype=None, hf_quantizer=None, is_safetensors=False, keep_in_fp32_modules=None, unexpected_keys=None):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    error_msgs = []
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    for param_name, param in state_dict.items():
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue
        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix):]
        module_name = param_name
        set_module_kwargs = {}
        if dtype is not None and torch.is_floating_point(param):
            if keep_in_fp32_modules is not None and any((module_to_keep_in_fp32 in param_name.split('.') for module_to_keep_in_fp32 in keep_in_fp32_modules)) and (dtype == torch.float16):
                param = param.to(torch.float32)
                if 'dtype' in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs['dtype'] = torch.float32
            else:
                param = param.to(dtype)
        old_param = model
        splits = param_name.split('.')
        for split in splits:
            old_param = getattr(old_param, split)
            if old_param is None:
                break
        if old_param is not None:
            if dtype is None:
                param = param.to(old_param.dtype)
            if old_param.is_contiguous():
                param = param.contiguous()
        set_module_kwargs['value'] = param
        if device_map is None:
            param_device = 'cpu'
        else:
            while len(module_name) > 0 and module_name not in device_map:
                module_name = '.'.join(module_name.split('.')[:-1])
            if module_name == '' and '' not in device_map:
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]
        if param_device == 'disk':
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == 'cpu' and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, model, state_dict_folder, state_dict_index)
        elif hf_quantizer is None or not hf_quantizer.requires_parameters_quantization or (not hf_quantizer.check_quantized_param(model, param, param_name, state_dict)):
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            hf_quantizer.create_quantized_param(model, param, param_name, param_device, state_dict, unexpected_keys)
    return (error_msgs, offload_index, state_dict_index)