import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
@dataclass
class SizedGenerator:
    """A generator that has a __len__ and can repeatedly call the generator
    function.
    """
    get_items: Callable[[], Generator]
    length: int

    def __len__(self):
        return self.length

    def __iter__(self):
        yield from self.get_items()