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
def _prepare_revert_transform(es, working_tree, target_tree, tt, filenames, backups, pp, basis_tree=None, merge_modified=None):
    with ui.ui_factory.nested_progress_bar() as child_pb:
        if merge_modified is None:
            merge_modified = working_tree.merge_modified()
        merge_modified = _alter_files(es, working_tree, target_tree, tt, child_pb, filenames, backups, merge_modified, basis_tree)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        raw_conflicts = resolve_conflicts(tt, child_pb, lambda t, c: conflict_pass(t, c, target_tree))
    conflicts = tt.cook_conflicts(raw_conflicts)
    return (conflicts, merge_modified)