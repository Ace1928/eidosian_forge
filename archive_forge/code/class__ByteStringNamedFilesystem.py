import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _ByteStringNamedFilesystem(Feature):
    """Is the filesystem based on bytes?"""

    def _probe(self):
        if os.name == 'posix':
            return True
        return False