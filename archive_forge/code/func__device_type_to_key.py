from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
import sys
def _device_type_to_key(device_type: str) -> str:
    if device_type == 'default':
        return 'CompositeExplicitAutograd'
    return torch._C._dispatch_key_for_device(device_type)