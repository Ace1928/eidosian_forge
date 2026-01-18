from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
class CompiledCursor_PositionIndexValue(ProtocolBuffer.ProtocolMessage):
    has_property_ = 0
    property_ = ''
    has_value_ = 0

    def __init__(self, contents=None):
        self.value_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.PropertyValue()
        if contents is not None:
            self.MergeFromString(contents)

    def property(self):
        return self.property_

    def set_property(self, x):
        self.has_property_ = 1
        self.property_ = x

    def clear_property(self):
        if self.has_property_:
            self.has_property_ = 0
            self.property_ = ''

    def has_property(self):
        return self.has_property_

    def value(self):
        return self.value_

    def mutable_value(self):
        self.has_value_ = 1
        return self.value_

    def clear_value(self):
        self.has_value_ = 0
        self.value_.Clear()

    def has_value(self):
        return self.has_value_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_property():
            self.set_property(x.property())
        if x.has_value():
            self.mutable_value().MergeFrom(x.value())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_property_ != x.has_property_:
            return 0
        if self.has_property_ and self.property_ != x.property_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        elif not self.value_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_property_:
            n += 2 + self.lengthString(len(self.property_))
        n += self.lengthString(self.value_.ByteSize())
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_property_:
            n += 2 + self.lengthString(len(self.property_))
        if self.has_value_:
            n += 2
            n += self.lengthString(self.value_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_property()
        self.clear_value()

    def OutputUnchecked(self, out):
        if self.has_property_:
            out.putVarInt32(242)
            out.putPrefixedString(self.property_)
        out.putVarInt32(250)
        out.putVarInt32(self.value_.ByteSize())
        self.value_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_property_:
            out.putVarInt32(242)
            out.putPrefixedString(self.property_)
        if self.has_value_:
            out.putVarInt32(250)
            out.putVarInt32(self.value_.ByteSizePartial())
            self.value_.OutputPartial(out)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 236:
                break
            if tt == 242:
                self.set_property(d.getPrefixedString())
                continue
            if tt == 250:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_value().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_property_:
            res += prefix + 'property: %s\n' % self.DebugFormatString(self.property_)
        if self.has_value_:
            res += prefix + 'value <\n'
            res += self.value_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res