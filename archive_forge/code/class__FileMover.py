import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
class _FileMover:
    """Moves and deletes files for TreeTransform, tracking operations"""

    def __init__(self):
        self.past_renames = []
        self.pending_deletions = []

    def rename(self, from_, to):
        """Rename a file from one path to another."""
        try:
            os.rename(from_, to)
        except OSError as e:
            if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExists(to, str(e))
            raise TransformRenameFailed(from_, to, str(e), e.errno)
        self.past_renames.append((from_, to))

    def pre_delete(self, from_, to):
        """Rename a file out of the way and mark it for deletion.

        Unlike os.unlink, this works equally well for files and directories.
        :param from_: The current file path
        :param to: A temporary path for the file
        """
        self.rename(from_, to)
        self.pending_deletions.append(to)

    def rollback(self):
        """Reverse all renames that have been performed"""
        for from_, to in reversed(self.past_renames):
            try:
                os.rename(to, from_)
            except OSError as e:
                raise TransformRenameFailed(to, from_, str(e), e.errno)
        self.past_renames = None
        self.pending_deletions = None

    def apply_deletions(self):
        """Apply all marked deletions"""
        for path in self.pending_deletions:
            delete_any(path)
        self.past_renames = None
        self.pending_deletions = None