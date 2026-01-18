import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _AddPathToDict(self, pbxbuildfile, path):
    """Adds path to the dict tracking paths belonging to this build phase.

    If the path is already a member of this build phase, raises an exception.
    """
    if path in self._files_by_path:
        raise ValueError('Found multiple build files with path ' + path)
    self._files_by_path[path] = pbxbuildfile