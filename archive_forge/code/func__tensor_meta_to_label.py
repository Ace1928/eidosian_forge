import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _tensor_meta_to_label(self, tm) -> str:
    if tm is None:
        return ''
    elif isinstance(tm, TensorMetadata):
        return self._stringify_tensor_meta(tm)
    elif isinstance(tm, list):
        result = ''
        for item in tm:
            result += self._tensor_meta_to_label(item)
        return result
    elif isinstance(tm, dict):
        result = ''
        for v in tm.values():
            result += self._tensor_meta_to_label(v)
        return result
    elif isinstance(tm, tuple):
        result = ''
        for item in tm:
            result += self._tensor_meta_to_label(item)
        return result
    else:
        raise RuntimeError(f'Unsupported tensor meta type {type(tm)}')