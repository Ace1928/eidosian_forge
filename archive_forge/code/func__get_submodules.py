import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def _get_submodules(model, key):
    parent = model.get_submodule('.'.join(key.split('.')[:-1]))
    target_name = key.split('.')[-1]
    target = model.get_submodule(key)
    return (parent, target, target_name)