from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
class FunctionPassManager(PassManager):

    def __init__(self, module):
        ptr = ffi.lib.LLVMPY_CreateFunctionPassManager(module)
        self._module = module
        module._owned = True
        PassManager.__init__(self, ptr)

    def initialize(self):
        """
        Initialize the FunctionPassManager.  Returns True if it produced
        any changes (?).
        """
        return ffi.lib.LLVMPY_InitializeFunctionPassManager(self)

    def finalize(self):
        """
        Finalize the FunctionPassManager.  Returns True if it produced
        any changes (?).
        """
        return ffi.lib.LLVMPY_FinalizeFunctionPassManager(self)

    def run(self, function, remarks_file=None, remarks_format='yaml', remarks_filter=''):
        """
        Run optimization passes on the given function.

        Parameters
        ----------
        function : llvmlite.binding.FunctionRef
            The function to be optimized inplace
        remarks_file : str; optional
            If not `None`, it is the file to store the optimization remarks.
        remarks_format : str; optional
            The format of the remarks file; the default is YAML
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
        if remarks_file is None:
            return ffi.lib.LLVMPY_RunFunctionPassManager(self, function)
        else:
            r = ffi.lib.LLVMPY_RunFunctionPassManagerWithRemarks(self, function, _encode_string(remarks_format), _encode_string(remarks_filter), _encode_string(remarks_file))
            if r == -1:
                raise IOError('Failed to initialize remarks file.')
            return bool(r)

    def run_with_remarks(self, function, remarks_format='yaml', remarks_filter=''):
        """
        Run optimization passes on the given function and returns the result
        and the remarks data.

        Parameters
        ----------
        function : llvmlite.binding.FunctionRef
            The function to be optimized inplace
        remarks_format : str; optional
            The format of the remarks file; the default is YAML
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
        remarkdesc, remarkfile = mkstemp()
        try:
            with os.fdopen(remarkdesc, 'r'):
                pass
            r = self.run(function, remarkfile, remarks_format, remarks_filter)
            if r == -1:
                raise IOError('Failed to initialize remarks file.')
            with open(remarkfile) as f:
                return (bool(r), f.read())
        finally:
            os.unlink(remarkfile)