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
def _create_log_revision_iterator(self):
    """Create a revision iterator for log.

        :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
            delta).
        """
    start_rev_id, end_rev_id = _get_revision_limits(self.branch, self.start_revision, self.end_revision)
    if self._match_using_deltas:
        return _log_revision_iterator_using_delta_matching(self.branch, delta_type=self.delta_type, match=self.match, levels=self.levels, specific_files=self.specific_files, start_rev_id=start_rev_id, end_rev_id=end_rev_id, direction=self.direction, exclude_common_ancestry=self.exclude_common_ancestry, limit=self.limit)
    else:
        file_count = len(self.specific_files)
        if file_count != 1:
            raise errors.BzrError('illegal LogRequest: must match-using-deltas when logging %d files' % file_count)
        return _log_revision_iterator_using_per_file_graph(self.branch, delta_type=self.delta_type, match=self.match, levels=self.levels, path=self.specific_files[0], start_rev_id=start_rev_id, end_rev_id=end_rev_id, direction=self.direction, exclude_common_ancestry=self.exclude_common_ancestry)