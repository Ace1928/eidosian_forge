import sys, os
import types
from . import model
from .error import VerificationError
def _get_declarations(self):
    lst = [(key, tp) for key, (tp, qual) in self.ffi._parser._declarations.items()]
    lst.sort()
    return lst