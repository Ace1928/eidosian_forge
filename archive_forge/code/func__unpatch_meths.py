import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _unpatch_meths(patchlist):
    for cls, name, old_meth in reversed(patchlist):
        setattr(cls, name, old_meth)