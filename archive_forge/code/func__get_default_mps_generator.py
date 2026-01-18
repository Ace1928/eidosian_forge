import torch
from .. import Tensor
from . import profiler
from .event import Event
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator