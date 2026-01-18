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
class _FakeGlobalNamespace:

    def __getattr__(self, name):
        if name == 'torch':
            return torch
        raise RuntimeError('Expected a torch namespace lookup')