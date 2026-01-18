import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def VerifyNoCollidingTargets(targets):
    """Verify that no two targets in the same directory share the same name.

  Arguments:
    targets: A list of targets in the form 'path/to/file.gyp:target_name'.
  """
    used = {}
    for target in targets:
        path, name = target.rsplit(':', 1)
        subdir, gyp = os.path.split(path)
        if not subdir:
            subdir = '.'
        key = subdir + ':' + name
        if key in used:
            raise GypError('Duplicate target name "%s" in directory "%s" used both in "%s" and "%s".' % (name, subdir, gyp, used[key]))
        used[key] = gyp