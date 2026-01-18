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
class Query_Filter(ProtocolBuffer.ProtocolMessage):
    LESS_THAN = 1
    LESS_THAN_OR_EQUAL = 2
    GREATER_THAN = 3
    GREATER_THAN_OR_EQUAL = 4
    EQUAL = 5
    IN = 6
    EXISTS = 7
    CONTAINED_IN_REGION = 8
    NOT_EQUAL = 9
    _Operator_NAMES = {1: 'LESS_THAN', 2: 'LESS_THAN_OR_EQUAL', 3: 'GREATER_THAN', 4: 'GREATER_THAN_OR_EQUAL', 5: 'EQUAL', 6: 'IN', 7: 'EXISTS', 8: 'CONTAINED_IN_REGION', 9: 'NOT_EQUAL'}

    def Operator_Name(cls, x):
        return cls._Operator_NAMES.get(x, '')
    Operator_Name = classmethod(Operator_Name)
    has_op_ = 0
    op_ = 0
    has_geo_region_ = 0
    geo_region_ = None

    def __init__(self, contents=None):
        self.property_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def op(self):
        return self.op_

    def set_op(self, x):
        self.has_op_ = 1
        self.op_ = x

    def clear_op(self):
        if self.has_op_:
            self.has_op_ = 0
            self.op_ = 0

    def has_op(self):
        return self.has_op_

    def property_size(self):
        return len(self.property_)

    def property_list(self):
        return self.property_

    def property(self, i):
        return self.property_[i]

    def mutable_property(self, i):
        return self.property_[i]

    def add_property(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Property()
        self.property_.append(x)
        return x

    def clear_property(self):
        self.property_ = []

    def geo_region(self):
        if self.geo_region_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.geo_region_ is None:
                    self.geo_region_ = GeoRegion()
            finally:
                self.lazy_init_lock_.release()
        return self.geo_region_

    def mutable_geo_region(self):
        self.has_geo_region_ = 1
        return self.geo_region()

    def clear_geo_region(self):
        if self.has_geo_region_:
            self.has_geo_region_ = 0
            if self.geo_region_ is not None:
                self.geo_region_.Clear()

    def has_geo_region(self):
        return self.has_geo_region_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_op():
            self.set_op(x.op())
        for i in range(x.property_size()):
            self.add_property().CopyFrom(x.property(i))
        if x.has_geo_region():
            self.mutable_geo_region().MergeFrom(x.geo_region())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_op_ != x.has_op_:
            return 0
        if self.has_op_ and self.op_ != x.op_:
            return 0
        if len(self.property_) != len(x.property_):
            return 0
        for e1, e2 in zip(self.property_, x.property_):
            if e1 != e2:
                return 0
        if self.has_geo_region_ != x.has_geo_region_:
            return 0
        if self.has_geo_region_ and self.geo_region_ != x.geo_region_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_op_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: op not set.')
        for p in self.property_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_geo_region_ and (not self.geo_region_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.op_)
        n += 1 * len(self.property_)
        for i in range(len(self.property_)):
            n += self.lengthString(self.property_[i].ByteSize())
        if self.has_geo_region_:
            n += 2 + self.lengthString(self.geo_region_.ByteSize())
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_op_:
            n += 1
            n += self.lengthVarInt64(self.op_)
        n += 1 * len(self.property_)
        for i in range(len(self.property_)):
            n += self.lengthString(self.property_[i].ByteSizePartial())
        if self.has_geo_region_:
            n += 2 + self.lengthString(self.geo_region_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_op()
        self.clear_property()
        self.clear_geo_region()

    def OutputUnchecked(self, out):
        out.putVarInt32(48)
        out.putVarInt32(self.op_)
        for i in range(len(self.property_)):
            out.putVarInt32(114)
            out.putVarInt32(self.property_[i].ByteSize())
            self.property_[i].OutputUnchecked(out)
        if self.has_geo_region_:
            out.putVarInt32(322)
            out.putVarInt32(self.geo_region_.ByteSize())
            self.geo_region_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_op_:
            out.putVarInt32(48)
            out.putVarInt32(self.op_)
        for i in range(len(self.property_)):
            out.putVarInt32(114)
            out.putVarInt32(self.property_[i].ByteSizePartial())
            self.property_[i].OutputPartial(out)
        if self.has_geo_region_:
            out.putVarInt32(322)
            out.putVarInt32(self.geo_region_.ByteSizePartial())
            self.geo_region_.OutputPartial(out)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 36:
                break
            if tt == 48:
                self.set_op(d.getVarInt32())
                continue
            if tt == 114:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_property().TryMerge(tmp)
                continue
            if tt == 322:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_geo_region().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_op_:
            res += prefix + 'op: %s\n' % self.DebugFormatInt32(self.op_)
        cnt = 0
        for e in self.property_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'property%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_geo_region_:
            res += prefix + 'geo_region <\n'
            res += self.geo_region_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res