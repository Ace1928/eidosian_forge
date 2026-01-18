import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _BackslashFilenameFeature(Feature):
    """Does the filesystem support backslashes in filenames?"""

    def _probe(self):
        try:
            fileno, name = tempfile.mkstemp(prefix='bzr\\prefix')
        except OSError:
            return False
        else:
            try:
                os.stat(name)
            except OSError:
                return False
            os.close(fileno)
            os.remove(name)
            return True