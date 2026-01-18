import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _write_source_to(self, file):
    self._vengine._f = file
    try:
        self._vengine.write_source_to_f()
    finally:
        del self._vengine._f