from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def DebugFormatFixed64(self, value):
    if value < 0:
        value += 1 << 64
    return '0x%x' % value