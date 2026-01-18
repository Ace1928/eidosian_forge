import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_struct(self, tp, name, module, **kwds):
    self._loaded_struct_or_union(tp)