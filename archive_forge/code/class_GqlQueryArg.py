from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class GqlQueryArg(ProtocolBuffer.ProtocolMessage):
    has_name_ = 0
    name_ = ''
    has_value_ = 0
    value_ = None
    has_cursor_ = 0
    cursor_ = ''

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def name(self):
        return self.name_

    def set_name(self, x):
        self.has_name_ = 1
        self.name_ = x

    def clear_name(self):
        if self.has_name_:
            self.has_name_ = 0
            self.name_ = ''

    def has_name(self):
        return self.has_name_

    def value(self):
        if self.value_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.value_ is None:
                    self.value_ = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Value()
            finally:
                self.lazy_init_lock_.release()
        return self.value_

    def mutable_value(self):
        self.has_value_ = 1
        return self.value()

    def clear_value(self):
        if self.has_value_:
            self.has_value_ = 0
            if self.value_ is not None:
                self.value_.Clear()

    def has_value(self):
        return self.has_value_

    def cursor(self):
        return self.cursor_

    def set_cursor(self, x):
        self.has_cursor_ = 1
        self.cursor_ = x

    def clear_cursor(self):
        if self.has_cursor_:
            self.has_cursor_ = 0
            self.cursor_ = ''

    def has_cursor(self):
        return self.has_cursor_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_name():
            self.set_name(x.name())
        if x.has_value():
            self.mutable_value().MergeFrom(x.value())
        if x.has_cursor():
            self.set_cursor(x.cursor())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_name_ != x.has_name_:
            return 0
        if self.has_name_ and self.name_ != x.name_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        if self.has_cursor_ != x.has_cursor_:
            return 0
        if self.has_cursor_ and self.cursor_ != x.cursor_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_value_ and (not self.value_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_name_:
            n += 1 + self.lengthString(len(self.name_))
        if self.has_value_:
            n += 1 + self.lengthString(self.value_.ByteSize())
        if self.has_cursor_:
            n += 1 + self.lengthString(len(self.cursor_))
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_name_:
            n += 1 + self.lengthString(len(self.name_))
        if self.has_value_:
            n += 1 + self.lengthString(self.value_.ByteSizePartial())
        if self.has_cursor_:
            n += 1 + self.lengthString(len(self.cursor_))
        return n

    def Clear(self):
        self.clear_name()
        self.clear_value()
        self.clear_cursor()

    def OutputUnchecked(self, out):
        if self.has_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.name_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putVarInt32(self.value_.ByteSize())
            self.value_.OutputUnchecked(out)
        if self.has_cursor_:
            out.putVarInt32(26)
            out.putPrefixedString(self.cursor_)

    def OutputPartial(self, out):
        if self.has_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.name_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putVarInt32(self.value_.ByteSizePartial())
            self.value_.OutputPartial(out)
        if self.has_cursor_:
            out.putVarInt32(26)
            out.putPrefixedString(self.cursor_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_name(d.getPrefixedString())
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_value().TryMerge(tmp)
                continue
            if tt == 26:
                self.set_cursor(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_name_:
            res += prefix + 'name: %s\n' % self.DebugFormatString(self.name_)
        if self.has_value_:
            res += prefix + 'value <\n'
            res += self.value_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_cursor_:
            res += prefix + 'cursor: %s\n' % self.DebugFormatString(self.cursor_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kname = 1
    kvalue = 2
    kcursor = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'name', 2: 'value', 3: 'cursor'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.STRING}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.GqlQueryArg'