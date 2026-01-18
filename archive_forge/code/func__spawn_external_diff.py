import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def _spawn_external_diff(diffcmd, capture_errors=True):
    """Spawn the external diff process, and return the child handle.

    :param diffcmd: The command list to spawn
    :param capture_errors: Capture stderr as well as setting LANG=C
        and LC_ALL=C. This lets us read and understand the output of diff,
        and respond to any errors.
    :return: A Popen object.
    """
    if capture_errors:
        env = {}
        path = os.environ.get('PATH')
        if path is not None:
            env['PATH'] = path
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        stderr = subprocess.PIPE
    else:
        env = None
        stderr = None
    try:
        pipe = subprocess.Popen(diffcmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=env)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise errors.NoDiff(str(e))
        raise
    return pipe