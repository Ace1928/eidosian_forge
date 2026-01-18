import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _write_packet(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
    packet = [SIGNATURE]
    packet.append(b'FF')
    packet.append(b'')
    flags = 8192
    if timestamp is not None:
        flags = flags | FLAG_TIMESTAMP
        since_epoch = timestamp - EPOCH
        nanoseconds = since_epoch.microseconds * 1000
        seconds = since_epoch.seconds + since_epoch.days * 24 * 3600
        packet.append(struct.pack(FMT_32, seconds))
        self._write_number(nanoseconds, packet)
    if test_id is not None:
        flags = flags | FLAG_TEST_ID
        self._write_utf8(test_id, packet)
    if test_tags:
        flags = flags | FLAG_TAGS
        self._write_number(len(test_tags), packet)
        for tag in test_tags:
            self._write_utf8(tag, packet)
    if runnable:
        flags = flags | FLAG_RUNNABLE
    if mime_type:
        flags = flags | FLAG_MIME_TYPE
        self._write_utf8(mime_type, packet)
    if file_name is not None:
        flags = flags | FLAG_FILE_CONTENT
        self._write_utf8(file_name, packet)
        self._write_number(len(file_bytes), packet)
        packet.append(file_bytes)
    if eof:
        flags = flags | FLAG_EOF
    if route_code is not None:
        flags = flags | FLAG_ROUTE_CODE
        self._write_utf8(route_code, packet)
    flags = flags | self.status_mask[test_status]
    packet[1] = struct.pack(FMT_16, flags)
    base_length = sum(map(len, packet)) + 4
    if base_length <= 62:
        length_length = 1
    elif base_length <= 16381:
        length_length = 2
    elif base_length <= 4194300:
        length_length = 3
    else:
        raise ValueError('Length too long: %r' % base_length)
    packet[2:3] = self._encode_number(base_length + length_length)
    content = b''.join(packet)
    data = content + struct.pack(FMT_32, zlib.crc32(content) & 4294967295)
    view = memoryview(data)
    datalen = len(data)
    offset = 0
    while offset < datalen:
        written = self.output_stream.write(view[offset:])
        if written is None:
            break
        offset += written
    self.output_stream.flush()