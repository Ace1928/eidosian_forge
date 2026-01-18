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
@staticmethod
def _fenc():
    """Returns safe encoding for passing file path to diff tool"""
    if sys.platform == 'win32':
        return 'mbcs'
    else:
        return sys.getfilesystemencoding() or 'ascii'