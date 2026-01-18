import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
def _nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass
    signatures.append(inspect.signature(nonzero))

    def nonzero(self, *, as_tuple: bool):
        pass
    signatures.append(inspect.signature(nonzero))
    return signatures