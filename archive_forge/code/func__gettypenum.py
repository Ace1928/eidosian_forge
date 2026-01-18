import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _gettypenum(self, type):
    return self._typesdict[type]