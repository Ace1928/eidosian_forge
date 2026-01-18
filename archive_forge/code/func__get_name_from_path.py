import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _get_name_from_path(self, path):
    if path == self._top_level_dir:
        return '.'
    path = _jython_aware_splitext(os.path.normpath(path))
    _relpath = os.path.relpath(path, self._top_level_dir)
    assert not os.path.isabs(_relpath), 'Path must be within the project'
    assert not _relpath.startswith('..'), 'Path must be within the project'
    name = _relpath.replace(os.path.sep, '.')
    return name