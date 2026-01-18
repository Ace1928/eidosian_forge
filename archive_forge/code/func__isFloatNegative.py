from __future__ import absolute_import
import array
import struct
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
def _isFloatNegative(self, value, encoded):
    if value == 0:
        return encoded[0] == 128
    return value < 0