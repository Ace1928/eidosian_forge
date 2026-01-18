from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_dot_postdom_printer_pass(self, show_body=False):
    """
        See https://llvm.org/docs/Passes.html#dot-postdom-print-postdominance-tree-of-function-to-dot-file

        LLVM 14: `llvm::createPostDomPrinterPass` and `llvm::createPostDomOnlyPrinterPass`
        """
    ffi.lib.LLVMPY_AddDotPostDomPrinterPass(self, show_body)