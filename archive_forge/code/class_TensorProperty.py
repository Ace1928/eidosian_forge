import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
class TensorProperty(enum.Enum):
    SIZE = 0
    STRIDE = 1
    STORAGE_OFFSET = 2

    def method_name(self):
        if self is TensorProperty.SIZE:
            return 'size'
        elif self is TensorProperty.STRIDE:
            return 'stride'
        elif self is TensorProperty.STORAGE_OFFSET:
            return 'storage_offset'