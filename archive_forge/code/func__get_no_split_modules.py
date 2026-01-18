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
def _get_no_split_modules(self, device_map: str):
    """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
    _no_split_modules = set()
    modules_to_check = [self]
    while len(modules_to_check) > 0:
        module = modules_to_check.pop(-1)
        if module.__class__.__name__ not in _no_split_modules:
            if isinstance(module, PreTrainedModel):
                if module._no_split_modules is None:
                    raise ValueError(f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model class needs to implement the `_no_split_modules` attribute.")
                else:
                    _no_split_modules = _no_split_modules | set(module._no_split_modules)
            modules_to_check += list(module.children())
    return list(_no_split_modules)