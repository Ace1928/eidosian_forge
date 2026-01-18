import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _BackslashDirSeparatorFeature(Feature):

    def _probe(self):
        try:
            os.lstat(os.getcwd() + '\\')
        except OSError:
            return False
        else:
            return True

    def feature_name(self):
        return "Filesystem treats '\\' as a directory separator."