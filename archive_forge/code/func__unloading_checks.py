from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Union
import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
def _unloading_checks(self, adapter_names: Optional[list[str]]):
    adapters_to_consider = adapter_names or self.active_adapters
    is_modules_to_save_available = any((self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider))
    if is_modules_to_save_available and len(adapters_to_consider) > 1:
        raise ValueError('Cannot unload multiple adapters that specify `modules_to_save`.')