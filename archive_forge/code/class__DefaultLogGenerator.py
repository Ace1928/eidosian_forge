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
class _DefaultLogGenerator(LogGenerator):
    """The default generator of log revisions."""

    def __init__(self, branch, levels=None, limit=None, diff_type=None, delta_type=None, show_signature=None, omit_merges=None, generate_tags=None, specific_files=None, match=None, start_revision=None, end_revision=None, direction=None, exclude_common_ancestry=None, _match_using_deltas=None, signature=None):
        self.branch = branch
        self.levels = levels
        self.limit = limit
        self.diff_type = diff_type
        self.delta_type = delta_type
        self.show_signature = signature
        self.omit_merges = omit_merges
        self.specific_files = specific_files
        self.match = match
        self.start_revision = start_revision
        self.end_revision = end_revision
        self.direction = direction
        self.exclude_common_ancestry = exclude_common_ancestry
        self._match_using_deltas = _match_using_deltas
        if generate_tags and branch.supports_tags():
            self.rev_tag_dict = branch.tags.get_reverse_tag_dict()
        else:
            self.rev_tag_dict = {}

    def iter_log_revisions(self):
        """Iterate over LogRevision objects.

        :return: An iterator yielding LogRevision objects.
        """
        log_count = 0
        revision_iterator = self._create_log_revision_iterator()
        for revs in revision_iterator:
            for (rev_id, revno, merge_depth), rev, delta in revs:
                if self.levels != 0 and merge_depth is not None and (merge_depth >= self.levels):
                    continue
                if self.omit_merges and len(rev.parent_ids) > 1:
                    continue
                if rev is None:
                    raise errors.GhostRevisionUnusableHere(rev_id)
                if self.diff_type is None:
                    diff = None
                else:
                    diff = _format_diff(self.branch, rev, self.diff_type, self.specific_files)
                if self.show_signature:
                    signature = format_signature_validity(rev_id, self.branch)
                else:
                    signature = None
                yield LogRevision(rev, revno, merge_depth, delta, self.rev_tag_dict.get(rev_id), diff, signature)
                if self.limit:
                    log_count += 1
                    if log_count >= self.limit:
                        return

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