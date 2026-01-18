import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
class device:

    def __new__(cls, device: _device_t):
        raise NotImplementedError()