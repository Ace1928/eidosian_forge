import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _write_utf8(self, a_string, packet):
    utf8 = a_string.encode('utf-8')
    self._write_number(len(utf8), packet)
    packet.append(utf8)