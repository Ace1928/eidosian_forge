import sys, os
import types
from . import model
from .error import VerificationError
def _prnt(self, what=''):
    self._f.write(what + '\n')