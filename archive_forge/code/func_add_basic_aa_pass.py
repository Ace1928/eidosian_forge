from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_basic_aa_pass(self):
    """
        See https://llvm.org/docs/Passes.html#basic-aa-basic-alias-analysis-stateless-aa-impl

        LLVM 14: `llvm::createBasicAAWrapperPass`
        """
    ffi.lib.LLVMPY_AddBasicAAWrapperPass(self)