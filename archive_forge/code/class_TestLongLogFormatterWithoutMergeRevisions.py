import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLongLogFormatterWithoutMergeRevisions(TestCaseForLogFormatter):

    def test_long_verbose_log(self):
        """Verbose log includes changed files

        bug #4676
        """
        wt = self.make_standard_commit('test_long_verbose_log', authors=[])
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_long_verbose_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\nadded:\n  a\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1), show_log_kwargs=dict(verbose=True))

    def test_long_verbose_contain_deltas(self):
        wt = self.make_branch_and_tree('parent')
        self.build_tree(['parent/f1', 'parent/f2'])
        wt.add(['f1', 'f2'])
        self.wt_commit(wt, 'first post')
        child_wt = wt.controldir.sprout('child').open_workingtree()
        os.unlink('child/f1')
        self.build_tree_contents([('child/f2', b'hello\n')])
        self.wt_commit(child_wt, 'removed f1 and modified f2')
        wt.merge_from_branch(child_wt.branch)
        self.wt_commit(wt, 'merge branch 1')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 2 [merge]\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  merge branch 1\nremoved:\n  f1\nmodified:\n  f2\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  first post\nadded:\n  f1\n  f2\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1), show_log_kwargs=dict(verbose=True))

    def test_long_trailing_newlines(self):
        wt = self.make_branch_and_tree('.')
        b = self.make_commits_with_trailing_newlines(wt)
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 3\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  single line with trailing newline\n------------------------------------------------------------\nrevno: 2\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:01 +0000\nmessage:\n  multiline\n  log\n  message\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  simple log message\n', b, log.LongLogFormatter, formatter_kwargs=dict(levels=1))

    def test_long_author_in_log(self):
        """Log includes the author name if it's set in
        the revision properties
        """
        wt = self.make_standard_commit('test_author_log')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_author_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1))

    def test_long_properties_in_log(self):
        """Log includes the custom properties returned by the registered
        handlers.
        """
        wt = self.make_standard_commit('test_properties_in_log')

        def trivial_custom_prop_handler(revision):
            return {'test_prop': 'test_value'}
        log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ntest_prop: test_value\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_properties_in_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=1))