import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
@staticmethod
def _extend_pythonpath(env):
    """Prepend current working dir to PATH environment variable if needed.

        If sys.path[0] is an empty string, the interpreter was likely
        invoked with -m and the effective path is about to change on
        re-exec.  Add the current directory to $PYTHONPATH to ensure
        that the new process sees the same path.

        This issue cannot be addressed in the general case because
        Python cannot reliably reconstruct the
        original command line (http://bugs.python.org/issue14208).

        (This idea filched from tornado.autoreload)
        """
    path_prefix = '.' + os.pathsep
    existing_path = env.get('PYTHONPATH', '')
    needs_patch = sys.path[0] == '' and (not existing_path.startswith(path_prefix))
    if needs_patch:
        env['PYTHONPATH'] = path_prefix + existing_path