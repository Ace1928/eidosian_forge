import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLogForAuthors(TestCaseForLogFormatter):

    def setUp(self):
        super().setUp()
        self.wt = self.make_standard_commit('nicky', authors=['John Doe <jdoe@example.com>', 'Jane Rey <jrey@example.com>'])

    def assertFormatterResult(self, formatter, who, result):
        formatter_kwargs = dict()
        if who is not None:
            author_list_handler = log.author_list_registry.get(who)
            formatter_kwargs['author_list_handler'] = author_list_handler
        TestCaseForLogFormatter.assertFormatterResult(self, result, self.wt.branch, formatter, formatter_kwargs=formatter_kwargs)

    def test_line_default(self):
        self.assertFormatterResult(log.LineLogFormatter, None, b'1: John Doe 2005-11-22 add a\n')

    def test_line_committer(self):
        self.assertFormatterResult(log.LineLogFormatter, 'committer', b'1: Lorem Ipsum 2005-11-22 add a\n')

    def test_line_first(self):
        self.assertFormatterResult(log.LineLogFormatter, 'first', b'1: John Doe 2005-11-22 add a\n')

    def test_line_all(self):
        self.assertFormatterResult(log.LineLogFormatter, 'all', b'1: John Doe, Jane Rey 2005-11-22 add a\n')

    def test_short_default(self):
        self.assertFormatterResult(log.ShortLogFormatter, None, b'    1 John Doe\t2005-11-22\n      add a\n\n')

    def test_short_committer(self):
        self.assertFormatterResult(log.ShortLogFormatter, 'committer', b'    1 Lorem Ipsum\t2005-11-22\n      add a\n\n')

    def test_short_first(self):
        self.assertFormatterResult(log.ShortLogFormatter, 'first', b'    1 John Doe\t2005-11-22\n      add a\n\n')

    def test_short_all(self):
        self.assertFormatterResult(log.ShortLogFormatter, 'all', b'    1 John Doe, Jane Rey\t2005-11-22\n      add a\n\n')

    def test_long_default(self):
        self.assertFormatterResult(log.LongLogFormatter, None, b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>, Jane Rey <jrey@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: nicky\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n')

    def test_long_committer(self):
        self.assertFormatterResult(log.LongLogFormatter, 'committer', b'------------------------------------------------------------\nrevno: 1\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: nicky\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n')

    def test_long_first(self):
        self.assertFormatterResult(log.LongLogFormatter, 'first', b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: nicky\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n')

    def test_long_all(self):
        self.assertFormatterResult(log.LongLogFormatter, 'all', b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>, Jane Rey <jrey@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: nicky\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n')

    def test_gnu_changelog_default(self):
        self.assertFormatterResult(log.GnuChangelogLogFormatter, None, b'2005-11-22  John Doe  <jdoe@example.com>\n\n\tadd a\n\n')

    def test_gnu_changelog_committer(self):
        self.assertFormatterResult(log.GnuChangelogLogFormatter, 'committer', b'2005-11-22  Lorem Ipsum  <test@example.com>\n\n\tadd a\n\n')

    def test_gnu_changelog_first(self):
        self.assertFormatterResult(log.GnuChangelogLogFormatter, 'first', b'2005-11-22  John Doe  <jdoe@example.com>\n\n\tadd a\n\n')

    def test_gnu_changelog_all(self):
        self.assertFormatterResult(log.GnuChangelogLogFormatter, 'all', b'2005-11-22  John Doe  <jdoe@example.com>, Jane Rey  <jrey@example.com>\n\n\tadd a\n\n')