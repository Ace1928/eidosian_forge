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
def _all_children(self, trans_id):
    children = self._all_children_cache.get(trans_id)
    if children is not None:
        return children
    children = set(self._transform.iter_tree_children(trans_id))
    children.difference_update(self._transform._new_parent)
    children.update(self._by_parent.get(trans_id, []))
    self._all_children_cache[trans_id] = children
    return children