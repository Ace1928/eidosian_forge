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
def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
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
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    error_msgs = []

    def load(module: nn.Module, state_dict, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + '.')
    load(model_to_load, state_dict, prefix=start_prefix)
    del state_dict
    return error_msgs