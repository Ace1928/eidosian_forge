from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueHttpHeader(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_value_ = 0
    value_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def key(self):
        return self.key_

    def set_key(self, x):
        self.has_key_ = 1
        self.key_ = x

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            self.key_ = ''

    def has_key(self):
        return self.has_key_

    def value(self):
        return self.value_

    def set_value(self, x):
        self.has_value_ = 1
        self.value_ = x

    def clear_value(self):
        if self.has_value_:
            self.has_value_ = 0
            self.value_ = ''

    def has_value(self):
        return self.has_value_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_value():
            self.set_value(x.value())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        n += self.lengthString(len(self.value_))
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_value_:
            n += 1
            n += self.lengthString(len(self.value_))
        return n

    def Clear(self):
        self.clear_key()
        self.clear_value()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.key_)
        out.putVarInt32(18)
        out.putPrefixedString(self.value_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(10)
            out.putPrefixedString(self.key_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putPrefixedString(self.value_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_value(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_value_:
            res += prefix + 'value: %s\n' % self.DebugFormatString(self.value_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kvalue = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'value'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueHttpHeader'