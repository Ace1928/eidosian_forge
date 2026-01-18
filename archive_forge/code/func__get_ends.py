import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
def _get_ends(self) -> Ints1d:
    return self._get_starts_ends()[1:]