import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_noop(self, tp, name, module, **kwds):
    pass