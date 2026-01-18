import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _get_info_for_log_files(revisionspec_list, file_list, exit_stack):
    """Find files and kinds given a list of files and a revision range.

    We search for files at the end of the range. If not found there,
    we try the start of the range.

    :param revisionspec_list: revision range as parsed on the command line
    :param file_list: the list of paths given on the command line;
      the first of these can be a branch location or a file path,
      the remainder must be file paths
    :param exit_stack: When the branch returned is read locked,
      an unlock call will be queued to the exit stack.
    :return: (branch, info_list, start_rev_info, end_rev_info) where
      info_list is a list of (relative_path, found, kind) tuples where
      kind is one of values 'directory', 'file', 'symlink', 'tree-reference'.
      branch will be read-locked.
    """
    from breezy.builtins import _get_revision_range
    tree, b, path = controldir.ControlDir.open_containing_tree_or_branch(file_list[0])
    exit_stack.enter_context(b.lock_read())
    if tree:
        relpaths = [path] + tree.safe_relpath_files(file_list[1:])
    else:
        relpaths = [path] + file_list[1:]
    info_list = []
    start_rev_info, end_rev_info = _get_revision_range(revisionspec_list, b, 'log')
    if relpaths in ([], ['']):
        return (b, [], start_rev_info, end_rev_info)
    if start_rev_info is None and end_rev_info is None:
        if tree is None:
            tree = b.basis_tree()
        tree1 = None
        for fp in relpaths:
            kind = _get_kind_for_file(tree, fp)
            if not kind:
                if tree1 is None:
                    try:
                        rev1 = b.get_rev_id(1)
                    except errors.NoSuchRevision:
                        kind = None
                    else:
                        tree1 = b.repository.revision_tree(rev1)
                if tree1:
                    kind = _get_kind_for_file(tree1, fp)
            info_list.append((fp, kind))
    elif start_rev_info == end_rev_info:
        tree = b.repository.revision_tree(end_rev_info.rev_id)
        for fp in relpaths:
            kind = _get_kind_for_file(tree, fp)
            info_list.append((fp, kind))
    else:
        rev_id = end_rev_info.rev_id
        if rev_id is None:
            tree = b.basis_tree()
        else:
            tree = b.repository.revision_tree(rev_id)
        tree1 = None
        for fp in relpaths:
            kind = _get_kind_for_file(tree, fp)
            if not kind:
                if tree1 is None:
                    rev_id = start_rev_info.rev_id
                    if rev_id is None:
                        rev1 = b.get_rev_id(1)
                        tree1 = b.repository.revision_tree(rev1)
                    else:
                        tree1 = b.repository.revision_tree(rev_id)
                kind = _get_kind_for_file(tree1, fp)
            info_list.append((fp, kind))
    return (b, info_list, start_rev_info, end_rev_info)