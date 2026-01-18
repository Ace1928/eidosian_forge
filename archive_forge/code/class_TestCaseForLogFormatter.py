import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestCaseForLogFormatter(tests.TestCaseWithTransport, TestLogMixin):

    def setUp(self):
        super().setUp()
        self.properties_handler_registry = log.properties_handler_registry
        log.properties_handler_registry = registry.Registry()

        def restore():
            log.properties_handler_registry = self.properties_handler_registry
        self.addCleanup(restore)

    def assertFormatterResult(self, result, branch, formatter_class, formatter_kwargs=None, show_log_kwargs=None):
        logfile = self.make_utf8_encoded_stringio()
        if formatter_kwargs is None:
            formatter_kwargs = {}
        formatter = formatter_class(to_file=logfile, **formatter_kwargs)
        if show_log_kwargs is None:
            show_log_kwargs = {}
        log.show_log(branch, formatter, **show_log_kwargs)
        self.assertEqualDiff(result, logfile.getvalue())

    def make_standard_commit(self, branch_nick, **kwargs):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        self.build_tree(['a'])
        wt.add(['a'])
        wt.branch.nick = branch_nick
        kwargs.setdefault('committer', 'Lorem Ipsum <test@example.com>')
        kwargs.setdefault('authors', ['John Doe <jdoe@example.com>'])
        self.wt_commit(wt, 'add a', **kwargs)
        return wt

    def make_commits_with_trailing_newlines(self, wt):
        """Helper method for LogFormatter tests"""
        b = wt.branch
        b.nick = 'test'
        self.build_tree_contents([('a', b'hello moto\n')])
        self.wt_commit(wt, 'simple log message', rev_id=b'a1')
        self.build_tree_contents([('b', b'goodbye\n')])
        wt.add('b')
        self.wt_commit(wt, 'multiline\nlog\nmessage\n', rev_id=b'a2')
        self.build_tree_contents([('c', b'just another manic monday\n')])
        wt.add('c')
        self.wt_commit(wt, 'single line with trailing newline\n', rev_id=b'a3')
        return b

    def _prepare_tree_with_merges(self, with_tags=False):
        wt = self.make_branch_and_memory_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        wt.add('')
        self.wt_commit(wt, 'rev-1', rev_id=b'rev-1')
        self.wt_commit(wt, 'rev-merged', rev_id=b'rev-2a')
        wt.set_parent_ids([b'rev-1', b'rev-2a'])
        wt.branch.set_last_revision_info(1, b'rev-1')
        self.wt_commit(wt, 'rev-2', rev_id=b'rev-2b')
        if with_tags:
            branch = wt.branch
            branch.tags.set_tag('v0.2', b'rev-2b')
            self.wt_commit(wt, 'rev-3', rev_id=b'rev-3')
            branch.tags.set_tag('v1.0rc1', b'rev-3')
            branch.tags.set_tag('v1.0', b'rev-3')
        return wt