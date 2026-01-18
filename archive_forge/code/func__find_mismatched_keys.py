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
def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys, add_prefix_to_model, remove_prefix_from_model, ignore_mismatched_sizes):
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            if checkpoint_key not in state_dict:
                continue
            model_key = checkpoint_key
            if remove_prefix_from_model:
                model_key = f'{prefix}.{checkpoint_key}'
            elif add_prefix_to_model:
                model_key = '.'.join(checkpoint_key.split('.')[1:])
            if model_key in model_state_dict and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape:
                if state_dict[checkpoint_key].shape[-1] == 1 and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel():
                    pass
                else:
                    mismatched_keys.append((checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape))
                    del state_dict[checkpoint_key]
    return mismatched_keys