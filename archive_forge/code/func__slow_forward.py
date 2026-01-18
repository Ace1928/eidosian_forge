from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
def _slow_forward(self, *input, **kwargs):
    tracing_state = torch._C._get_tracing_state()
    if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
        return self.forward(*input, **kwargs)
    recording_scopes = torch.jit._trace._trace_module_map is not None
    if recording_scopes:
        name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None
        if name:
            tracing_state.push_scope(name)
        else:
            recording_scopes = False
    try:
        result = self.forward(*input, **kwargs)
    finally:
        if recording_scopes:
            tracing_state.pop_scope()
    return result