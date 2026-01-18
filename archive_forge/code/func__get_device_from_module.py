import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def _get_device_from_module(module: str):
    if module.split('.')[-1] in ['cuda', torch._C._get_privateuse1_backend_name()]:
        return module.split('.')[-1]
    else:
        return 'cpu'