import sys, os
import types
from . import model
from .error import VerificationError
def _load_known_int_constant(self, module, funcname):
    BType = self.ffi._typeof_locked('char[]')[0]
    BFunc = self.ffi._typeof_locked('int(*)(char*)')[0]
    function = module.load_function(BFunc, funcname)
    p = self.ffi.new(BType, 256)
    if function(p) < 0:
        error = self.ffi.string(p)
        if sys.version_info >= (3,):
            error = str(error, 'utf-8')
        raise VerificationError(error)