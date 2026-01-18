from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _check_open_dir(self, fs, path, info):
    """Check if a directory should be considered in the walk."""
    if self.exclude_dirs is not None and fs.match(self.exclude_dirs, info.name):
        return False
    if self.filter_dirs is not None and (not fs.match(self.filter_dirs, info.name)):
        return False
    return self.check_open_dir(fs, path, info)