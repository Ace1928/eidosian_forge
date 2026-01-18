from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def ErrorCode_Name(cls, x):
    return cls._ErrorCode_NAMES.get(x, '')