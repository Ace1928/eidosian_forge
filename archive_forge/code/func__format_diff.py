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
def _format_diff(branch, rev, diff_type, files=None):
    """Format a diff.

    :param branch: Branch object
    :param rev: Revision object
    :param diff_type: Type of diff to generate
    :param files: List of files to generate diff for (or None for all)
    """
    repo = branch.repository
    if len(rev.parent_ids) == 0:
        ancestor_id = _mod_revision.NULL_REVISION
    else:
        ancestor_id = rev.parent_ids[0]
    tree_1 = repo.revision_tree(ancestor_id)
    tree_2 = repo.revision_tree(rev.revision_id)
    if diff_type == 'partial' and files is not None:
        specific_files = files
    else:
        specific_files = None
    s = BytesIO()
    path_encoding = get_diff_header_encoding()
    diff.show_diff_trees(tree_1, tree_2, s, specific_files, old_label='', new_label='', path_encoding=path_encoding)
    return s.getvalue()