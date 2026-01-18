import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _get_available_device_type():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    return None