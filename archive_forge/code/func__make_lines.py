import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
@staticmethod
def _make_lines():
    lines = []
    for group in range(48):
        offset = group * 50
        numbers = list(range(offset, offset + 50))
        lines.append(b''.join((b'%d' % n for n in numbers)) + b'\n')
    return lines