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
class DiffTreeReference(DiffPath):

    def diff(self, old_path, new_path, old_kind, new_kind):
        """Perform comparison between two tree references.  (dummy)

        """
        if 'tree-reference' not in (old_kind, new_kind):
            return self.CANNOT_DIFF
        if old_kind not in ('tree-reference', None):
            return self.CANNOT_DIFF
        if new_kind not in ('tree-reference', None):
            return self.CANNOT_DIFF
        return self.CHANGED