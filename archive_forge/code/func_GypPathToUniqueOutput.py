import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def GypPathToUniqueOutput(self, path, qualified=True):
    """Translate a gyp path to a ninja path for writing output.

        If qualified is True, qualify the resulting filename with the name
        of the target.  This is necessary when e.g. compiling the same
        path twice for two separate output targets.

        See the above discourse on path conversions."""
    path = self.ExpandSpecial(path)
    assert not path.startswith('$'), path
    obj = 'obj'
    if self.toolset != 'target':
        obj += '.' + self.toolset
    path_dir, path_basename = os.path.split(path)
    assert not os.path.isabs(path_dir), "'%s' can not be absolute path (see crbug.com/462153)." % path_dir
    if qualified:
        path_basename = self.name + '.' + path_basename
    return os.path.normpath(os.path.join(obj, self.base_dir, path_dir, path_basename))