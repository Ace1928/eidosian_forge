import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _AttribFeature(Feature):

    def _probe(self):
        if sys.platform not in ('cygwin', 'win32'):
            return False
        try:
            proc = subprocess.Popen(['attrib', '.'], stdout=subprocess.PIPE)
        except OSError:
            return False
        return 0 == proc.wait()

    def feature_name(self):
        return 'attrib Windows command-line tool'