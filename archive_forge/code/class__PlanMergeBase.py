import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
class _PlanMergeBase:

    def __init__(self, a_rev, b_rev, vf, key_prefix):
        """Contructor.

        :param a_rev: Revision-id of one revision to merge
        :param b_rev: Revision-id of the other revision to merge
        :param vf: A VersionedFiles containing both revisions
        :param key_prefix: A prefix for accessing keys in vf, typically
            (file_id,).
        """
        self.a_rev = a_rev
        self.b_rev = b_rev
        self.vf = vf
        self._last_lines = None
        self._last_lines_revision_id = None
        self._cached_matching_blocks = {}
        self._key_prefix = key_prefix
        self._precache_tip_lines()

    def _precache_tip_lines(self):
        lines = self.get_lines([self.a_rev, self.b_rev])
        self.lines_a = lines[self.a_rev]
        self.lines_b = lines[self.b_rev]

    def get_lines(self, revisions):
        """Get lines for revisions from the backing VersionedFiles.

        :raises RevisionNotPresent: on absent texts.
        """
        keys = [self._key_prefix + (rev,) for rev in revisions]
        result = {}
        for record in self.vf.get_record_stream(keys, 'unordered', True):
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(record.key, self.vf)
            result[record.key[-1]] = record.get_bytes_as('lines')
        return result

    def plan_merge(self):
        """Generate a 'plan' for merging the two revisions.

        This involves comparing their texts and determining the cause of
        differences.  If text A has a line and text B does not, then either the
        line was added to text A, or it was deleted from B.  Once the causes
        are combined, they are written out in the format described in
        VersionedFile.plan_merge
        """
        blocks = self._get_matching_blocks(self.a_rev, self.b_rev)
        unique_a, unique_b = self._unique_lines(blocks)
        new_a, killed_b = self._determine_status(self.a_rev, unique_a)
        new_b, killed_a = self._determine_status(self.b_rev, unique_b)
        return self._iter_plan(blocks, new_a, killed_b, new_b, killed_a)

    def _iter_plan(self, blocks, new_a, killed_b, new_b, killed_a):
        last_i = 0
        last_j = 0
        for i, j, n in blocks:
            for a_index in range(last_i, i):
                if a_index in new_a:
                    if a_index in killed_b:
                        yield ('conflicted-a', self.lines_a[a_index])
                    else:
                        yield ('new-a', self.lines_a[a_index])
                else:
                    yield ('killed-b', self.lines_a[a_index])
            for b_index in range(last_j, j):
                if b_index in new_b:
                    if b_index in killed_a:
                        yield ('conflicted-b', self.lines_b[b_index])
                    else:
                        yield ('new-b', self.lines_b[b_index])
                else:
                    yield ('killed-a', self.lines_b[b_index])
            for a_index in range(i, i + n):
                yield ('unchanged', self.lines_a[a_index])
            last_i = i + n
            last_j = j + n

    def _get_matching_blocks(self, left_revision, right_revision):
        """Return a description of which sections of two revisions match.

        See SequenceMatcher.get_matching_blocks
        """
        cached = self._cached_matching_blocks.get((left_revision, right_revision))
        if cached is not None:
            return cached
        if self._last_lines_revision_id == left_revision:
            left_lines = self._last_lines
            right_lines = self.get_lines([right_revision])[right_revision]
        else:
            lines = self.get_lines([left_revision, right_revision])
            left_lines = lines[left_revision]
            right_lines = lines[right_revision]
        self._last_lines = right_lines
        self._last_lines_revision_id = right_revision
        matcher = patiencediff.PatienceSequenceMatcher(None, left_lines, right_lines)
        return matcher.get_matching_blocks()

    def _unique_lines(self, matching_blocks):
        """Analyse matching_blocks to determine which lines are unique

        :return: a tuple of (unique_left, unique_right), where the values are
            sets of line numbers of unique lines.
        """
        last_i = 0
        last_j = 0
        unique_left = []
        unique_right = []
        for i, j, n in matching_blocks:
            unique_left.extend(range(last_i, i))
            unique_right.extend(range(last_j, j))
            last_i = i + n
            last_j = j + n
        return (unique_left, unique_right)

    @staticmethod
    def _subtract_plans(old_plan, new_plan):
        """Remove changes from new_plan that came from old_plan.

        It is assumed that the difference between the old_plan and new_plan
        is their choice of 'b' text.

        All lines from new_plan that differ from old_plan are emitted
        verbatim.  All lines from new_plan that match old_plan but are
        not about the 'b' revision are emitted verbatim.

        Lines that match and are about the 'b' revision are the lines we
        don't want, so we convert 'killed-b' -> 'unchanged', and 'new-b'
        is skipped entirely.
        """
        matcher = patiencediff.PatienceSequenceMatcher(None, old_plan, new_plan)
        last_j = 0
        for i, j, n in matcher.get_matching_blocks():
            for jj in range(last_j, j):
                yield new_plan[jj]
            for jj in range(j, j + n):
                plan_line = new_plan[jj]
                if plan_line[0] == 'new-b':
                    pass
                elif plan_line[0] == 'killed-b':
                    yield ('unchanged', plan_line[1])
                else:
                    yield plan_line
            last_j = j + n