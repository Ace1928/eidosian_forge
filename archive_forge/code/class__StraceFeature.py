import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _StraceFeature(Feature):

    def _probe(self):
        try:
            proc = subprocess.Popen(['strace'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            proc.communicate()
            return True
        except OSError as e:
            import errno
            if e.errno == errno.ENOENT:
                return False
            else:
                raise

    def feature_name(self):
        return 'strace'