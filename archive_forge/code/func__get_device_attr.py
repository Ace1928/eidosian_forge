import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == 'cuda':
        return get_member(torch.cuda)
    if device_type and device_type.lower() == 'xpu':
        return get_member(torch.xpu)
    if device_type == torch._C._get_privateuse1_backend_name():
        return get_member(getattr(torch, device_type))
    return None