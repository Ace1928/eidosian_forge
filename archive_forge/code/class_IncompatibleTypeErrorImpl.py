from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
class IncompatibleTypeErrorImpl(Exception):

    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = 'invalid operands of type ' + self.type_a.__repr__() + ' and ' + self.type_b.__repr__()
        super(IncompatibleTypeErrorImpl, self).__init__(self.message)