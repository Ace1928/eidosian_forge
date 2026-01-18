import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class SymlinkFeature(Feature):
    """Whether symlinks can be created by the current user."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _probe(self):
        return osutils.supports_symlinks(self.path)

    def feature_name(self):
        return 'symlinks'