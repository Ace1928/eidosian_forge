import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLineLogFormatterWithMergeRevisions(TestCaseForLogFormatter):

    def test_line_merge_revs_log(self):
        """Line log should show revno

        bug #5162
        """
        wt = self.make_standard_commit('test-line-log', committer='Line-Log-Formatter Tester <test@line.log>', authors=[])
        self.assertFormatterResult(b'1: Line-Log-Formatte... 2005-11-22 add a\n', wt.branch, log.LineLogFormatter)

    def test_line_merge_revs_log_single_merge_revision(self):
        wt = self._prepare_tree_with_merges()
        revspec = revisionspec.RevisionSpec.from_string('1.1.1')
        rev = revspec.in_history(wt.branch)
        self.assertFormatterResult(b'1.1.1: Joe Foo 2005-11-22 rev-merged\n', wt.branch, log.LineLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(start_revision=rev, end_revision=rev))

    def test_line_merge_revs_log_with_merges(self):
        wt = self._prepare_tree_with_merges()
        self.assertFormatterResult(b'2: Joe Foo 2005-11-22 [merge] rev-2\n  1.1.1: Joe Foo 2005-11-22 rev-merged\n1: Joe Foo 2005-11-22 rev-1\n', wt.branch, log.LineLogFormatter, formatter_kwargs=dict(levels=0))