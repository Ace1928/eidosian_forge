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
class Query_Order(ProtocolBuffer.ProtocolMessage):
    ASCENDING = 1
    DESCENDING = 2
    _Direction_NAMES = {1: 'ASCENDING', 2: 'DESCENDING'}

    def Direction_Name(cls, x):
        return cls._Direction_NAMES.get(x, '')
    Direction_Name = classmethod(Direction_Name)
    has_property_ = 0
    property_ = ''
    has_direction_ = 0
    direction_ = 1

    def __init__(self, contents=None):
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

    def direction(self):
        return self.direction_

    def set_direction(self, x):
        self.has_direction_ = 1
        self.direction_ = x

    def clear_direction(self):
        if self.has_direction_:
            self.has_direction_ = 0
            self.direction_ = 1

    def has_direction(self):
        return self.has_direction_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_property():
            self.set_property(x.property())
        if x.has_direction():
            self.set_direction(x.direction())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_property_ != x.has_property_:
            return 0
        if self.has_property_ and self.property_ != x.property_:
            return 0
        if self.has_direction_ != x.has_direction_:
            return 0
        if self.has_direction_ and self.direction_ != x.direction_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_property_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: property not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.property_))
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_property_:
            n += 1
            n += self.lengthString(len(self.property_))
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        return n

    def Clear(self):
        self.clear_property()
        self.clear_direction()

    def OutputUnchecked(self, out):
        out.putVarInt32(82)
        out.putPrefixedString(self.property_)
        if self.has_direction_:
            out.putVarInt32(88)
            out.putVarInt32(self.direction_)

    def OutputPartial(self, out):
        if self.has_property_:
            out.putVarInt32(82)
            out.putPrefixedString(self.property_)
        if self.has_direction_:
            out.putVarInt32(88)
            out.putVarInt32(self.direction_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 76:
                break
            if tt == 82:
                self.set_property(d.getPrefixedString())
                continue
            if tt == 88:
                self.set_direction(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_property_:
            res += prefix + 'property: %s\n' % self.DebugFormatString(self.property_)
        if self.has_direction_:
            res += prefix + 'direction: %s\n' % self.DebugFormatInt32(self.direction_)
        return res