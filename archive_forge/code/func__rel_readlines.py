import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _rel_readlines(self, filename):
    g = open(os.path.join(os.path.dirname(__file__), filename), 'r')
    lines = g.readlines()
    g.close()
    return lines