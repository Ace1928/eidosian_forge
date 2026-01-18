import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _BreakinFeature(Feature):
    """Does this platform support the breakin feature?"""

    def _probe(self):
        from breezy import breakin
        if breakin.determine_signal() is None:
            return False
        if sys.platform == 'win32':
            try:
                import ctypes
            except OSError:
                return False
        return True

    def feature_name(self):
        return 'SIGQUIT or SIGBREAK w/ctypes on win32'