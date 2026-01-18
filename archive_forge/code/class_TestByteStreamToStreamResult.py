import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
class TestByteStreamToStreamResult(TestCase):

    def test_non_subunit_encapsulated(self):
        source = BytesIO(b'foo\nbar\n')
        result = StreamResult()
        subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
        self.assertEqual([('status', None, None, None, True, 'stdout', b'f', False, None, None, None), ('status', None, None, None, True, 'stdout', b'o', False, None, None, None), ('status', None, None, None, True, 'stdout', b'o', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\n', False, None, None, None), ('status', None, None, None, True, 'stdout', b'b', False, None, None, None), ('status', None, None, None, True, 'stdout', b'a', False, None, None, None), ('status', None, None, None, True, 'stdout', b'r', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\n', False, None, None, None)], result._events)
        self.assertEqual(b'', source.read())

    def test_signature_middle_utf8_char(self):
        utf8_bytes = b'\xe3\xb3\x8a'
        source = BytesIO(utf8_bytes)
        result = StreamResult()
        subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
        self.assertEqual([('status', None, None, None, True, 'stdout', b'\xe3', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\xb3', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\x8a', False, None, None, None)], result._events)

    def test_non_subunit_disabled_raises(self):
        source = BytesIO(b'foo\nbar\n')
        result = StreamResult()
        case = subunit.ByteStreamToStreamResult(source)
        e = self.assertRaises(Exception, case.run, result)
        self.assertEqual(b'f', e.args[1])
        self.assertEqual(b'oo\nbar\n', source.read())
        self.assertEqual([], result._events)

    def test_trivial_enumeration(self):
        source = BytesIO(CONSTANT_ENUM)
        result = StreamResult()
        subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
        self.assertEqual(b'', source.read())
        self.assertEqual([('status', 'foo', 'exists', None, True, None, None, False, None, None, None)], result._events)

    def test_multiple_events(self):
        source = BytesIO(CONSTANT_ENUM + CONSTANT_ENUM)
        result = StreamResult()
        subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
        self.assertEqual(b'', source.read())
        self.assertEqual([('status', 'foo', 'exists', None, True, None, None, False, None, None, None), ('status', 'foo', 'exists', None, True, None, None, False, None, None, None)], result._events)

    def test_inprogress(self):
        self.check_event(CONSTANT_INPROGRESS, 'inprogress')

    def test_success(self):
        self.check_event(CONSTANT_SUCCESS, 'success')

    def test_uxsuccess(self):
        self.check_event(CONSTANT_UXSUCCESS, 'uxsuccess')

    def test_skip(self):
        self.check_event(CONSTANT_SKIP, 'skip')

    def test_fail(self):
        self.check_event(CONSTANT_FAIL, 'fail')

    def test_xfail(self):
        self.check_event(CONSTANT_XFAIL, 'xfail')

    def check_events(self, source_bytes, events):
        source = BytesIO(source_bytes)
        result = StreamResult()
        subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
        self.assertEqual(b'', source.read())
        self.assertEqual(events, result._events)
        for event in result._events:
            if event[5] is not None:
                bytes(event[6])

    def check_event(self, source_bytes, test_status=None, test_id='foo', route_code=None, timestamp=None, tags=None, mime_type=None, file_name=None, file_bytes=None, eof=False, runnable=True):
        event = self._event(test_id=test_id, test_status=test_status, tags=tags, runnable=runnable, file_name=file_name, file_bytes=file_bytes, eof=eof, mime_type=mime_type, route_code=route_code, timestamp=timestamp)
        self.check_events(source_bytes, [event])

    def _event(self, test_status=None, test_id=None, route_code=None, timestamp=None, tags=None, mime_type=None, file_name=None, file_bytes=None, eof=False, runnable=True):
        return ('status', test_id, test_status, tags, runnable, file_name, file_bytes, eof, mime_type, route_code, timestamp)

    def test_eof(self):
        self.check_event(CONSTANT_EOF, test_id=None, eof=True)

    def test_file_content(self):
        self.check_event(CONSTANT_FILE_CONTENT, test_id=None, file_name='barney', file_bytes=b'woo')

    def test_file_content_length_into_checksum(self):
        bad_file_length_content = b'\xb3!@\x13\x06barney\x04woo\xdc\xe2\xdb5'
        self.check_events(bad_file_length_content, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=bad_file_length_content, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'File content extends past end of packet: claimed 4 bytes, 3 available', mime_type='text/plain;charset=utf8')])

    def test_packet_length_4_word_varint(self):
        packet_data = b'\xb3!@\xc0\x00\x11'
        self.check_events(packet_data, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=packet_data, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'3 byte maximum given but 4 byte value found.', mime_type='text/plain;charset=utf8')])

    def test_mime(self):
        self.check_event(CONSTANT_MIME, test_id=None, mime_type='application/foo; charset=1')

    def test_route_code(self):
        self.check_event(CONSTANT_ROUTE_CODE, 'success', route_code='source', test_id='bar')

    def test_runnable(self):
        self.check_event(CONSTANT_RUNNABLE, test_status='success', runnable=False)

    def test_tags(self):
        self.check_event(CONSTANT_TAGS[0], None, tags={'foo', 'bar'}, test_id='bar')

    def test_timestamp(self):
        timestamp = datetime.datetime(2001, 12, 12, 12, 59, 59, 45, iso8601.UTC)
        self.check_event(CONSTANT_TIMESTAMP, 'success', test_id='bar', timestamp=timestamp)

    def test_bad_crc_errors_via_status(self):
        file_bytes = CONSTANT_MIME[:-1] + b'\x00'
        self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'Bad checksum - calculated (0x78335115), stored (0x78335100)', mime_type='text/plain;charset=utf8')])

    def test_not_utf8_in_string(self):
        file_bytes = CONSTANT_ROUTE_CODE[:5] + b'\xb4' + CONSTANT_ROUTE_CODE[6:-4] + b'\xceV\xc6\x17'
        self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'UTF8 string at offset 2 is not UTF8', mime_type='text/plain;charset=utf8')])

    def test_NULL_in_string(self):
        file_bytes = CONSTANT_ROUTE_CODE[:6] + b'\x00' + CONSTANT_ROUTE_CODE[7:-4] + b'\xd7A\xac\xfe'
        self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'UTF8 string at offset 2 contains NUL byte', mime_type='text/plain;charset=utf8')])

    def test_bad_utf8_stringlength(self):
        file_bytes = CONSTANT_ROUTE_CODE[:4] + b'?' + CONSTANT_ROUTE_CODE[5:-4] + b'\xbe)\xe0\xc2'
        self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'UTF8 string at offset 2 extends past end of packet: claimed 63 bytes, 10 available', mime_type='text/plain;charset=utf8')])

    def test_route_code_and_file_content(self):
        content = BytesIO()
        subunit.StreamResultToBytes(content).status(route_code='0', mime_type='text/plain', file_name='bar', file_bytes=b'foo')
        self.check_event(content.getvalue(), test_id=None, file_name='bar', route_code='0', mime_type='text/plain', file_bytes=b'foo')
    if st is not None:

        @given(st.binary())
        def test_hypothesis_decoding(self, code_bytes):
            source = BytesIO(code_bytes)
            result = StreamResult()
            stream = subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout')
            stream.run(result)
            self.assertEqual(b'', source.read())