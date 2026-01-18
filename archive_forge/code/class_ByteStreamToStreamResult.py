import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
class ByteStreamToStreamResult(object):
    """Parse a subunit byte stream.

    Mixed streams that contain non-subunit content is supported when a
    non_subunit_name is passed to the contructor. The default is to raise an
    error containing the non-subunit byte after it has been read from the
    stream.

    Typical use:

       >>> case = ByteStreamToStreamResult(sys.stdin.buffer)
       >>> result = StreamResult()
       >>> result.startTestRun()
       >>> case.run(result)
       >>> result.stopTestRun()
    """
    status_lookup = {0: None, 1: 'exists', 2: 'inprogress', 3: 'success', 4: 'uxsuccess', 5: 'skip', 6: 'fail', 7: 'xfail'}

    def __init__(self, source, non_subunit_name=None):
        """Create a ByteStreamToStreamResult.

        :param source: A file like object to read bytes from. Must support
            read(<count>) and return bytes. The file is not closed by
            ByteStreamToStreamResult. subunit.make_stream_binary() is
            called on the stream to get it into bytes mode.
        :param non_subunit_name: If set to non-None, non subunit content
            encountered in the stream will be converted into file packets
            labelled with this name.
        """
        self.non_subunit_name = non_subunit_name
        self.source = subunit.make_stream_binary(source)
        self.codec = codecs.lookup('utf8').incrementaldecoder()

    def run(self, result):
        """Parse source and emit events to result.

        This is a blocking call: it will run until EOF is detected on source.
        """
        self.codec.reset()
        mid_character = False
        while True:
            content = self.source.read(1)
            if not content:
                return
            if not mid_character and content[0] == SIGNATURE[0]:
                self._parse_packet(result)
                continue
            if self.non_subunit_name is None:
                raise Exception('Non subunit content', content)
            try:
                if self.codec.decode(content):
                    mid_character = False
                else:
                    mid_character = True
            except UnicodeDecodeError:
                mid_character = False
            buffered = [content]
            while len(buffered[-1]):
                if sys.platform == 'win32':
                    break
                try:
                    self.source.fileno()
                except:
                    break
                readable = select.select([self.source], [], [], 1e-06)[0]
                if readable:
                    content = self.source.read(1)
                    if not len(content):
                        break
                    if not mid_character and content[0] == SIGNATURE[0]:
                        break
                    buffered.append(content)
                    try:
                        if self.codec.decode(content):
                            mid_character = False
                        else:
                            mid_character = True
                    except UnicodeDecodeError:
                        mid_character = False
                if not readable or len(buffered) >= 1048576:
                    break
            result.status(file_name=self.non_subunit_name, file_bytes=b''.join(buffered))
            if mid_character or not len(content) or content[0] != SIGNATURE[0]:
                continue
            self._parse_packet(result)

    def _parse_packet(self, result):
        try:
            packet = [SIGNATURE]
            self._parse(packet, result)
        except ParseError as error:
            result.status(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=b''.join(packet), mime_type='application/octet-stream')
            result.status(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=error.args[0].encode('utf8'), mime_type='text/plain;charset=utf8')

    def _parse_varint(self, data, pos, max_3_bytes=False):
        data_0 = struct.unpack(FMT_8, data[pos:pos + 1])[0]
        typeenum = data_0 & 192
        value_0 = data_0 & 63
        if typeenum == 0:
            return (value_0, 1)
        elif typeenum == 64:
            data_1 = struct.unpack(FMT_8, data[pos + 1:pos + 2])[0]
            return (value_0 << 8 | data_1, 2)
        elif typeenum == 128:
            data_1 = struct.unpack(FMT_16, data[pos + 1:pos + 3])[0]
            return (value_0 << 16 | data_1, 3)
        else:
            if max_3_bytes:
                raise ParseError('3 byte maximum given but 4 byte value found.')
            data_1, data_2 = struct.unpack(FMT_24, data[pos + 1:pos + 4])
            result = value_0 << 24 | data_1 << 8 | data_2
            return (result, 4)

    def _parse(self, packet, result):
        header = read_exactly(self.source, 5)
        packet.append(header)
        flags = struct.unpack(FMT_16, header[:2])[0]
        length, consumed = self._parse_varint(header, 2, max_3_bytes=True)
        remainder = read_exactly(self.source, length - 6)
        if consumed != 3:
            packet[-1] += remainder
            pos = 2 + consumed
        else:
            packet.append(remainder)
            pos = 0
        crc = zlib.crc32(packet[0])
        for fragment in packet[1:-1]:
            crc = zlib.crc32(fragment, crc)
        crc = zlib.crc32(packet[-1][:-4], crc) & 4294967295
        packet_crc = struct.unpack(FMT_32, packet[-1][-4:])[0]
        if crc != packet_crc:
            raise ParseError('Bad checksum - calculated (0x%x), stored (0x%x)' % (crc, packet_crc))
        if hasattr(builtins, 'memoryview'):
            body = memoryview(packet[-1])
        else:
            body = packet[-1]
        body = body[:-4]
        if flags & FLAG_TIMESTAMP:
            seconds = struct.unpack(FMT_32, body[pos:pos + 4])[0]
            nanoseconds, consumed = self._parse_varint(body, pos + 4)
            pos = pos + 4 + consumed
            timestamp = EPOCH + datetime.timedelta(seconds=seconds, microseconds=nanoseconds / 1000)
        else:
            timestamp = None
        if flags & FLAG_TEST_ID:
            test_id, pos = self._read_utf8(body, pos)
        else:
            test_id = None
        if flags & FLAG_TAGS:
            tag_count, consumed = self._parse_varint(body, pos)
            pos += consumed
            test_tags = set()
            for _ in range(tag_count):
                tag, pos = self._read_utf8(body, pos)
                test_tags.add(tag)
        else:
            test_tags = None
        if flags & FLAG_MIME_TYPE:
            mime_type, pos = self._read_utf8(body, pos)
        else:
            mime_type = None
        if flags & FLAG_FILE_CONTENT:
            file_name, pos = self._read_utf8(body, pos)
            content_length, consumed = self._parse_varint(body, pos)
            pos += consumed
            file_bytes = body[pos:pos + content_length]
            if len(file_bytes) != content_length:
                raise ParseError('File content extends past end of packet: claimed %d bytes, %d available' % (content_length, len(file_bytes)))
            pos += content_length
        else:
            file_name = None
            file_bytes = None
        if flags & FLAG_ROUTE_CODE:
            route_code, pos = self._read_utf8(body, pos)
        else:
            route_code = None
        runnable = bool(flags & FLAG_RUNNABLE)
        eof = bool(flags & FLAG_EOF)
        test_status = self.status_lookup[flags & 7]
        result.status(test_id=test_id, test_status=test_status, test_tags=test_tags, runnable=runnable, mime_type=mime_type, eof=eof, file_name=file_name, file_bytes=file_bytes, route_code=route_code, timestamp=timestamp)
    __call__ = run

    def _read_utf8(self, buf, pos):
        length, consumed = self._parse_varint(buf, pos)
        pos += consumed
        utf8_bytes = buf[pos:pos + length]
        if length != len(utf8_bytes):
            raise ParseError('UTF8 string at offset %d extends past end of packet: claimed %d bytes, %d available' % (pos - 2, length, len(utf8_bytes)))
        if NUL_ELEMENT in utf8_bytes:
            raise ParseError('UTF8 string at offset %d contains NUL byte' % (pos - 2,))
        try:
            utf8, decoded_bytes = utf_8_decode(utf8_bytes)
            if decoded_bytes != length:
                raise ParseError('Invalid (partially decodable) string at offset %d, %d undecoded bytes' % (pos - 2, length - decoded_bytes))
            return (utf8, length + pos)
        except UnicodeDecodeError:
            raise ParseError('UTF8 string at offset %d is not UTF8' % (pos - 2,))