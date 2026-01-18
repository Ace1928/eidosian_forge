import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
class ExportEntry(object):
    """
    A simple record for exporting symbols.
    """

    def __init__(self, symbol, signature, function):
        self.symbol = symbol
        self.signature = signature
        self.function = function

    def __repr__(self):
        return 'ExportEntry(%r, %r)' % (self.symbol, self.signature)