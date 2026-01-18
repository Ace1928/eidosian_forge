from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def DebugFormatInt32(self, value):
    if value <= -2000000000 or value >= 2000000000:
        return self.DebugFormatFixed32(value)
    return '%d' % value