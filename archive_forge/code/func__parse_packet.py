import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _parse_packet(self, result):
    try:
        packet = [SIGNATURE]
        self._parse(packet, result)
    except ParseError as error:
        result.status(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=b''.join(packet), mime_type='application/octet-stream')
        result.status(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=error.args[0].encode('utf8'), mime_type='text/plain;charset=utf8')