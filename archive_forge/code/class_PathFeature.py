import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class PathFeature(Feature):
    """Feature testing whether a particular path exists."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _probe(self):
        return os.path.exists(self.path)

    def feature_name(self):
        return '%s exists' % self.path