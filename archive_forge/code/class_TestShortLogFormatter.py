import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestShortLogFormatter(TestCaseForLogFormatter):

    def test_trailing_newlines(self):
        wt = self.make_branch_and_tree('.')
        b = self.make_commits_with_trailing_newlines(wt)
        self.assertFormatterResult(b'    3 Joe Foo\t2005-11-22\n      single line with trailing newline\n\n    2 Joe Foo\t2005-11-22\n      multiline\n      log\n      message\n\n    1 Joe Foo\t2005-11-22\n      simple log message\n\n', b, log.ShortLogFormatter)

    def test_short_log_with_merges(self):
        wt = self._prepare_tree_with_merges()
        self.assertFormatterResult(b'    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\n', wt.branch, log.ShortLogFormatter)

    def test_short_log_with_merges_and_advice(self):
        wt = self._prepare_tree_with_merges()
        self.assertFormatterResult(b'    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\nUse --include-merged or -n0 to see merged revisions.\n', wt.branch, log.ShortLogFormatter, formatter_kwargs=dict(show_advice=True))

    def test_short_log_with_merges_and_range(self):
        wt = self._prepare_tree_with_merges()
        self.wt_commit(wt, 'rev-3a', rev_id=b'rev-3a')
        wt.branch.set_last_revision_info(2, b'rev-2b')
        wt.set_parent_ids([b'rev-2b', b'rev-3a'])
        self.wt_commit(wt, 'rev-3b', rev_id=b'rev-3b')
        self.assertFormatterResult(b'    3 Joe Foo\t2005-11-22 [merge]\n      rev-3b\n\n    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n', wt.branch, log.ShortLogFormatter, show_log_kwargs=dict(start_revision=2, end_revision=3))

    def test_short_log_with_tags(self):
        wt = self._prepare_tree_with_merges(with_tags=True)
        self.assertFormatterResult(b'    3 Joe Foo\t2005-11-22 {v1.0, v1.0rc1}\n      rev-3\n\n    2 Joe Foo\t2005-11-22 {v0.2} [merge]\n      rev-2\n\n    1 Joe Foo\t2005-11-22\n      rev-1\n\n', wt.branch, log.ShortLogFormatter)

    def test_short_log_single_merge_revision(self):
        wt = self._prepare_tree_with_merges()
        revspec = revisionspec.RevisionSpec.from_string('1.1.1')
        rev = revspec.in_history(wt.branch)
        self.assertFormatterResult(b'      1.1.1 Joe Foo\t2005-11-22\n            rev-merged\n\n', wt.branch, log.ShortLogFormatter, show_log_kwargs=dict(start_revision=rev, end_revision=rev))

    def test_show_ids(self):
        wt = self.make_branch_and_tree('parent')
        self.build_tree(['parent/f1', 'parent/f2'])
        wt.add(['f1', 'f2'])
        self.wt_commit(wt, 'first post', rev_id=b'a')
        child_wt = wt.controldir.sprout('child').open_workingtree()
        self.wt_commit(child_wt, 'branch 1 changes', rev_id=b'b')
        wt.merge_from_branch(child_wt.branch)
        self.wt_commit(wt, 'merge branch 1', rev_id=b'c')
        self.assertFormatterResult(b'    2 Joe Foo\t2005-11-22 [merge]\n      revision-id:c\n      merge branch 1\n\n          1.1.1 Joe Foo\t2005-11-22\n                revision-id:b\n                branch 1 changes\n\n    1 Joe Foo\t2005-11-22\n      revision-id:a\n      first post\n\n', wt.branch, log.ShortLogFormatter, formatter_kwargs=dict(levels=0, show_ids=True))