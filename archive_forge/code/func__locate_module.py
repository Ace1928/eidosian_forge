import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _locate_module(self):
    if not os.path.isfile(self.modulefilename):
        if self.ext_package:
            try:
                pkg = __import__(self.ext_package, None, None, ['__doc__'])
            except ImportError:
                return
            path = pkg.__path__
        else:
            path = None
        filename = self._vengine.find_module(self.get_module_name(), path, _get_so_suffixes())
        if filename is None:
            return
        self.modulefilename = filename
    self._vengine.collect_types()
    self._has_module = True