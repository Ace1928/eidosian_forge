from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
class ModulePassManager(PassManager):

    def __init__(self, ptr=None):
        if ptr is None:
            ptr = ffi.lib.LLVMPY_CreatePassManager()
        PassManager.__init__(self, ptr)

    def run(self, module, remarks_file=None, remarks_format='yaml', remarks_filter=''):
        """
        Run optimization passes on the given module.

        Parameters
        ----------
        module : llvmlite.binding.ModuleRef
            The module to be optimized inplace
        remarks_file : str; optional
            If not `None`, it is the file to store the optimization remarks.
        remarks_format : str; optional
            The format to write; YAML is default
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
        if remarks_file is None:
            return ffi.lib.LLVMPY_RunPassManager(self, module)
        else:
            r = ffi.lib.LLVMPY_RunPassManagerWithRemarks(self, module, _encode_string(remarks_format), _encode_string(remarks_filter), _encode_string(remarks_file))
            if r == -1:
                raise IOError('Failed to initialize remarks file.')
            return r > 0

    def run_with_remarks(self, module, remarks_format='yaml', remarks_filter=''):
        """
        Run optimization passes on the given module and returns the result and
        the remarks data.

        Parameters
        ----------
        module : llvmlite.binding.ModuleRef
            The module to be optimized
        remarks_format : str
            The remarks output; YAML is the default
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
        remarkdesc, remarkfile = mkstemp()
        try:
            with os.fdopen(remarkdesc, 'r'):
                pass
            r = self.run(module, remarkfile, remarks_format, remarks_filter)
            if r == -1:
                raise IOError('Failed to initialize remarks file.')
            with open(remarkfile) as f:
                return (bool(r), f.read())
        finally:
            os.unlink(remarkfile)