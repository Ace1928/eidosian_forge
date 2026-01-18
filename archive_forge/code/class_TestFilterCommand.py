import subprocess
import sys
import unittest
from datetime import datetime
from io import BytesIO
from testtools import TestCase
from testtools.compat import _b
from testtools.testresult.doubles import ExtendedTestResult, StreamResult
import iso8601
import subunit
from subunit.test_results import make_tag_filter, TestResultFilter
from subunit import ByteStreamToStreamResult, StreamResultToBytes
class TestFilterCommand(TestCase):

    def run_command(self, args, stream):
        command = [sys.executable, '-m', 'subunit.filter_scripts.subunit_filter'] + list(args)
        ps = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ps.communicate(stream)
        if ps.returncode != 0:
            raise RuntimeError('{} failed: {}'.format(command, err))
        return out

    def test_default(self):
        byte_stream = BytesIO()
        stream = StreamResultToBytes(byte_stream)
        stream.status(test_id='foo', test_status='inprogress')
        stream.status(test_id='foo', test_status='skip')
        output = self.run_command([], byte_stream.getvalue())
        events = StreamResult()
        ByteStreamToStreamResult(BytesIO(output)).run(events)
        {event[1] for event in events._events}
        self.assertEqual([('status', 'foo', 'inprogress'), ('status', 'foo', 'skip')], [event[:3] for event in events._events])

    def test_tags(self):
        byte_stream = BytesIO()
        stream = StreamResultToBytes(byte_stream)
        stream.status(test_id='foo', test_status='inprogress', test_tags={'a'})
        stream.status(test_id='foo', test_status='success', test_tags={'a'})
        stream.status(test_id='bar', test_status='inprogress')
        stream.status(test_id='bar', test_status='inprogress')
        stream.status(test_id='baz', test_status='inprogress', test_tags={'a'})
        stream.status(test_id='baz', test_status='success', test_tags={'a'})
        output = self.run_command(['-s', '--with-tag', 'a'], byte_stream.getvalue())
        events = StreamResult()
        ByteStreamToStreamResult(BytesIO(output)).run(events)
        ids = {event[1] for event in events._events}
        self.assertEqual({'foo', 'baz'}, ids)

    def test_no_passthrough(self):
        output = self.run_command(['--no-passthrough'], b'hi thar')
        self.assertEqual(b'', output)

    def test_passthrough(self):
        output = self.run_command([], b'hi thar')
        byte_stream = BytesIO()
        stream = StreamResultToBytes(byte_stream)
        stream.status(file_name='stdout', file_bytes=b'hi thar')
        self.assertEqual(byte_stream.getvalue(), output)