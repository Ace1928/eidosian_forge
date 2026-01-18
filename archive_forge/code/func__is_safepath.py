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
def _is_safepath(self, path):
    """Return true if `path` may be able to pass to subprocess."""
    fenc = self._fenc()
    try:
        return path == path.encode(fenc).decode(fenc)
    except UnicodeError:
        return False