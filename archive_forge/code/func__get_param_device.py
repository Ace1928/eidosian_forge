import contextlib
import gc
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import packaging
import torch
import torch.nn as nn
from ..state import AcceleratorState
from .constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import AutocastKwargs, CustomDtype, DistributedType
from .imports import is_mps_available, is_npu_available, is_peft_available, is_torch_xla_available, is_xpu_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index
from .tqdm import is_tqdm_available, tqdm
from .versions import compare_versions
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
def _get_param_device(param, device_map):
    if param in device_map:
        return device_map[param]
    parent_param = '.'.join(param.split('.')[:-1])
    if parent_param == param:
        raise ValueError(f'The `device_map` does not contain the module {param}.')
    else:
        return _get_param_device(parent_param, device_map)