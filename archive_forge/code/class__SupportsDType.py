from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):

    @property
    def dtype(self) -> _DType_co:
        ...