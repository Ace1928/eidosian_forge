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
class DiffKindChange:
    """Special differ for file kind changes.

    Represents kind change as deletion + creation.  Uses the other differs
    to do this.
    """

    def __init__(self, differs):
        self.differs = differs

    def finish(self):
        pass

    @classmethod
    def from_diff_tree(klass, diff_tree):
        return klass(diff_tree.differs)

    def diff(self, old_path, new_path, old_kind, new_kind):
        """Perform comparison

        :param old_path: Path of the file in the old tree
        :param new_path: Path of the file in the new tree
        :param old_kind: Old file-kind of the file
        :param new_kind: New file-kind of the file
        """
        if None in (old_kind, new_kind):
            return DiffPath.CANNOT_DIFF
        result = DiffPath._diff_many(self.differs, old_path, new_path, old_kind, None)
        if result is DiffPath.CANNOT_DIFF:
            return result
        return DiffPath._diff_many(self.differs, old_path, new_path, None, new_kind)