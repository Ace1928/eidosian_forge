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
def _all_available_adapter_names(self) -> list[str]:
    """Return a sorted list of all available adapter names"""
    adapter_names = set()
    for name in self.adapter_layer_names + self.other_param_names:
        attr = getattr(self, name)
        if hasattr(attr, 'keys'):
            adapter_names.update(attr.keys())
    return sorted(adapter_names)