import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
@staticmethod
def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
    raise NotImplementedError()