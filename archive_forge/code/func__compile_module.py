import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _compile_module(self):
    tmpdir = os.path.dirname(self.sourcefilename)
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
    try:
        same = ffiplatform.samefile(outputfilename, self.modulefilename)
    except OSError:
        same = False
    if not same:
        _ensure_dir(self.modulefilename)
        shutil.move(outputfilename, self.modulefilename)
    self._has_module = True