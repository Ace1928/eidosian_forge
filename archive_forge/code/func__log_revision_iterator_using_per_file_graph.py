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
def _log_revision_iterator_using_per_file_graph(branch, delta_type, match, levels, path, start_rev_id, end_rev_id, direction, exclude_common_ancestry):
    view_revisions = _calc_view_revisions(branch, start_rev_id, end_rev_id, direction, generate_merge_revisions=True, exclude_common_ancestry=exclude_common_ancestry)
    if not isinstance(view_revisions, list):
        view_revisions = list(view_revisions)
    view_revisions = _filter_revisions_touching_path(branch, path, view_revisions, include_merges=levels != 1)
    return make_log_rev_iterator(branch, view_revisions, delta_type, match)