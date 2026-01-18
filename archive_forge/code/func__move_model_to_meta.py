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
def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
    """
    Moves `loaded_state_dict_keys` in model to meta device which frees up the memory taken by those params.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    for k in loaded_state_dict_keys:
        submodule, param_name = find_submodule_and_param_name(model, k, start_prefix)
        if submodule is not None:
            new_val = getattr(submodule, param_name)
            if isinstance(new_val, torch.nn.Parameter):
                new_val = torch.nn.Parameter(new_val.to('meta'))
            else:
                new_val = new_val.to('meta')
            setattr(submodule, param_name, new_val)