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
def _diff_many(differs, old_path, new_path, old_kind, new_kind):
    for file_differ in differs:
        result = file_differ.diff(old_path, new_path, old_kind, new_kind)
        if result is not DiffPath.CANNOT_DIFF:
            return result
    else:
        return DiffPath.CANNOT_DIFF