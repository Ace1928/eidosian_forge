import datetime
import optparse
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
from tempfile import NamedTemporaryFile
from iso8601 import UTC
from testtools import TestCase
from testtools.matchers import (Equals, Matcher, MatchesAny, MatchesListwise,
from testtools.testresult.doubles import StreamResult
import subunit._output as _o
from subunit._output import (_ALL_ACTIONS, _FINAL_ACTIONS,
class TestStatusArgParserTests(TestCase):
    scenarios = [(cmd, dict(command=cmd, option='--' + cmd)) for cmd in _ALL_ACTIONS]

    def test_can_parse_all_commands_with_test_id(self):
        test_id = self.getUniqueString()
        args = safe_parse_arguments(args=[self.option, test_id])
        self.assertThat(args.action, Equals(self.command))
        self.assertThat(args.test_id, Equals(test_id))

    def test_all_commands_parse_file_attachment(self):
        with NamedTemporaryFile() as tmp_file:
            args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', tmp_file.name])
            self.assertThat(args.attach_file.name, Equals(tmp_file.name))

    def test_all_commands_accept_mimetype_argument(self):
        with NamedTemporaryFile() as tmp_file:
            args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', tmp_file.name, '--mimetype', 'text/plain'])
            self.assertThat(args.mimetype, Equals('text/plain'))

    def test_all_commands_accept_file_name_argument(self):
        with NamedTemporaryFile() as tmp_file:
            args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', tmp_file.name, '--file-name', 'foo'])
            self.assertThat(args.file_name, Equals('foo'))

    def test_all_commands_accept_tags_argument(self):
        args = safe_parse_arguments(args=[self.option, 'foo', '--tag', 'foo', '--tag', 'bar', '--tag', 'baz'])
        self.assertThat(args.tags, Equals(['foo', 'bar', 'baz']))

    def test_attach_file_with_hyphen_opens_stdin(self):
        self.patch(_o.sys, 'stdin', TextIOWrapper(BytesIO(b'Hello')))
        args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', '-'])
        self.assertThat(args.attach_file.read(), Equals(b'Hello'))

    def test_attach_file_with_hyphen_sets_filename_to_stdin(self):
        args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', '-'])
        self.assertThat(args.file_name, Equals('stdin'))

    def test_can_override_stdin_filename(self):
        args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', '-', '--file-name', 'foo'])
        self.assertThat(args.file_name, Equals('foo'))

    def test_requires_test_id(self):

        def fn():
            return safe_parse_arguments(args=[self.option])
        self.assertThat(fn, raises(RuntimeError('argument %s: must specify a single TEST_ID.' % self.option)))