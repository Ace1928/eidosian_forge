import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
class CopyIfCallgrind:
    """Signal that a global may be replaced with a deserialized copy.

    See `GlobalsBridge` for why this matters.
    """

    def __init__(self, value: Any, *, setup: Optional[str]=None):
        for method, supported_types in _GLOBALS_ALLOWED_TYPES.items():
            if any((isinstance(value, t) for t in supported_types)):
                self._value: Any = value
                self._setup: Optional[str] = setup
                self._serialization: Serialization = method
                break
        else:
            supported_str = '\n'.join([getattr(t, '__name__', repr(t)) for t in it.chain(_GLOBALS_ALLOWED_TYPES.values())])
            raise ValueError(f'Unsupported type: {type(value)}\n`collect_callgrind` restricts globals to the following types:\n{textwrap.indent(supported_str, '  ')}')

    @property
    def value(self) -> Any:
        return self._value

    @property
    def setup(self) -> Optional[str]:
        return self._setup

    @property
    def serialization(self) -> Serialization:
        return self._serialization

    @staticmethod
    def unwrap_all(globals: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, CopyIfCallgrind) else v for k, v in globals.items()}