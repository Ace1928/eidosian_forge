from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def add_const(self, const):
    """
        Add a constant to the environment, return its index.
        """
    if isinstance(const, str):
        const = sys.intern(const)
    for index, val in enumerate(self.env.consts):
        if val is const:
            break
    else:
        index = len(self.env.consts)
        self.env.consts.append(const)
    return index