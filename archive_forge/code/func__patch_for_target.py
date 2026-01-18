import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _patch_for_target(patchlist, target):
    from cffi._shimmed_dist_utils import build_ext
    if target.endswith('.*'):
        target = target[:-2]
        if sys.platform == 'win32':
            target += '.dll'
        elif sys.platform == 'darwin':
            target += '.dylib'
        else:
            target += '.so'
    _patch_meth(patchlist, build_ext, 'get_ext_filename', lambda self, ext_name: target)