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
class StatusStreamResultTests(TestCase):
    scenarios = [(s, dict(status=s, option='--' + s)) for s in _ALL_ACTIONS]
    _dummy_timestamp = datetime.datetime(2013, 1, 1, 0, 0, 0, 0, UTC)

    def setUp(self):
        super().setUp()
        self.patch(_o, 'create_timestamp', lambda: self._dummy_timestamp)
        self.test_id = self.getUniqueString()

    def test_only_one_packet_is_generated(self):
        result = get_result_for([self.option, self.test_id])
        self.assertThat(len(result._events), Equals(3))

    def test_correct_status_is_generated(self):
        result = get_result_for([self.option, self.test_id])
        self.assertThat(result._events[1], MatchesStatusCall(test_status=self.status))

    def test_all_commands_generate_tags(self):
        result = get_result_for([self.option, self.test_id, '--tag', 'hello', '--tag', 'world'])
        self.assertThat(result._events[1], MatchesStatusCall(test_tags={'hello', 'world'}))

    def test_all_commands_generate_timestamp(self):
        result = get_result_for([self.option, self.test_id])
        self.assertThat(result._events[1], MatchesStatusCall(timestamp=self._dummy_timestamp))

    def test_all_commands_generate_correct_test_id(self):
        result = get_result_for([self.option, self.test_id])
        self.assertThat(result._events[1], MatchesStatusCall(test_id=self.test_id))

    def test_file_is_sent_in_single_packet(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_can_read_binary_files(self):
        with temp_file_contents(b'\xde\xad\xbe\xef') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'\xde\xad\xbe\xef', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_can_read_empty_files(self):
        with temp_file_contents(b'') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'', file_name=f.name, eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_can_read_stdin(self):
        self.patch(_o.sys, 'stdin', TextIOWrapper(BytesIO(b'\xfe\xed\xfa\xce')))
        result = get_result_for([self.option, self.test_id, '--attach-file', '-'])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'\xfe\xed\xfa\xce', file_name='stdin', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_file_is_sent_with_test_id(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_file_is_sent_with_test_status(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_status=self.status, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_file_chunk_size_is_honored(self):
        with temp_file_contents(b'Hello') as f:
            self.patch(_o, '_CHUNK_SIZE', 1)
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, file_bytes=b'H', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'e', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'l', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'l', eof=False), MatchesStatusCall(test_id=self.test_id, file_bytes=b'o', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_file_mimetype_specified_once_only(self):
        with temp_file_contents(b'Hi') as f:
            self.patch(_o, '_CHUNK_SIZE', 1)
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name, '--mimetype', 'text/plain'])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, mime_type='text/plain', file_bytes=b'H', eof=False), MatchesStatusCall(test_id=self.test_id, mime_type=None, file_bytes=b'i', eof=True), MatchesStatusCall(call='stopTestRun')]))

    def test_tags_specified_once_only(self):
        with temp_file_contents(b'Hi') as f:
            self.patch(_o, '_CHUNK_SIZE', 1)
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name, '--tag', 'foo', '--tag', 'bar'])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, test_tags={'foo', 'bar'}), MatchesStatusCall(test_id=self.test_id, test_tags=None), MatchesStatusCall(call='stopTestRun')]))

    def test_timestamp_specified_once_only(self):
        with temp_file_contents(b'Hi') as f:
            self.patch(_o, '_CHUNK_SIZE', 1)
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(test_id=self.test_id, timestamp=self._dummy_timestamp), MatchesStatusCall(test_id=self.test_id, timestamp=None), MatchesStatusCall(call='stopTestRun')]))

    def test_test_status_specified_once_only(self):
        with temp_file_contents(b'Hi') as f:
            self.patch(_o, '_CHUNK_SIZE', 1)
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            if self.status in _FINAL_ACTIONS:
                first_call = MatchesStatusCall(test_id=self.test_id, test_status=None)
                last_call = MatchesStatusCall(test_id=self.test_id, test_status=self.status)
            else:
                first_call = MatchesStatusCall(test_id=self.test_id, test_status=self.status)
                last_call = MatchesStatusCall(test_id=self.test_id, test_status=None)
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), first_call, last_call, MatchesStatusCall(call='stopTestRun')]))

    def test_filename_can_be_overridden(self):
        with temp_file_contents(b'Hello') as f:
            specified_file_name = self.getUniqueString()
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name, '--file-name', specified_file_name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=specified_file_name, file_bytes=b'Hello'), MatchesStatusCall(call='stopTestRun')]))

    def test_file_name_is_used_by_default(self):
        with temp_file_contents(b'Hello') as f:
            result = get_result_for([self.option, self.test_id, '--attach-file', f.name])
            self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=f.name, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))