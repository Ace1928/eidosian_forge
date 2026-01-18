import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
def _ensure_tensor_on_device(self, inputs, device):
    if isinstance(inputs, ModelOutput):
        return ModelOutput({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
    elif isinstance(inputs, dict):
        return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
    elif isinstance(inputs, UserDict):
        return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
    elif isinstance(inputs, list):
        return [self._ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
    elif isinstance(inputs, torch.Tensor):
        if device == torch.device('cpu') and inputs.dtype in {torch.float16, torch.bfloat16}:
            inputs = inputs.float()
        return inputs.to(device)
    else:
        return inputs