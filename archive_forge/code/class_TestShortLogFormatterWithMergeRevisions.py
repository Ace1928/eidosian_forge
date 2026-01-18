import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestShortLogFormatterWithMergeRevisions(TestCaseForLogFormatter):

    def test_short_merge_revs_log_with_merges(self):
        wt = self._prepare_tree_with_merges()
        self.assertFormatterResult(b'    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n          1.1.1 Joe Foo\t2005-11-22\n                rev-merged\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\n', wt.branch, log.ShortLogFormatter, formatter_kwargs=dict(levels=0))

    def test_short_merge_revs_log_single_merge_revision(self):
        wt = self._prepare_tree_with_merges()
        revspec = revisionspec.RevisionSpec.from_string('1.1.1')
        rev = revspec.in_history(wt.branch)
        self.assertFormatterResult(b'      1.1.1 Joe Foo\t2005-11-22\n            rev-merged\n\n', wt.branch, log.ShortLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(start_revision=rev, end_revision=rev))