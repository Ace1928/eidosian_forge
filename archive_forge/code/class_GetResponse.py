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
class GetResponse(ProtocolBuffer.ProtocolMessage):
    has_in_order_ = 0
    in_order_ = 1

    def __init__(self, contents=None):
        self.entity_ = []
        self.deferred_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def entity_size(self):
        return len(self.entity_)

    def entity_list(self):
        return self.entity_

    def entity(self, i):
        return self.entity_[i]

    def mutable_entity(self, i):
        return self.entity_[i]

    def add_entity(self):
        x = GetResponse_Entity()
        self.entity_.append(x)
        return x

    def clear_entity(self):
        self.entity_ = []

    def deferred_size(self):
        return len(self.deferred_)

    def deferred_list(self):
        return self.deferred_

    def deferred(self, i):
        return self.deferred_[i]

    def mutable_deferred(self, i):
        return self.deferred_[i]

    def add_deferred(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
        self.deferred_.append(x)
        return x

    def clear_deferred(self):
        self.deferred_ = []

    def in_order(self):
        return self.in_order_

    def set_in_order(self, x):
        self.has_in_order_ = 1
        self.in_order_ = x

    def clear_in_order(self):
        if self.has_in_order_:
            self.has_in_order_ = 0
            self.in_order_ = 1

    def has_in_order(self):
        return self.has_in_order_

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.entity_size()):
            self.add_entity().CopyFrom(x.entity(i))
        for i in range(x.deferred_size()):
            self.add_deferred().CopyFrom(x.deferred(i))
        if x.has_in_order():
            self.set_in_order(x.in_order())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.entity_) != len(x.entity_):
            return 0
        for e1, e2 in zip(self.entity_, x.entity_):
            if e1 != e2:
                return 0
        if len(self.deferred_) != len(x.deferred_):
            return 0
        for e1, e2 in zip(self.deferred_, x.deferred_):
            if e1 != e2:
                return 0
        if self.has_in_order_ != x.has_in_order_:
            return 0
        if self.has_in_order_ and self.in_order_ != x.in_order_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.entity_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.deferred_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.entity_)
        for i in range(len(self.entity_)):
            n += self.entity_[i].ByteSize()
        n += 1 * len(self.deferred_)
        for i in range(len(self.deferred_)):
            n += self.lengthString(self.deferred_[i].ByteSize())
        if self.has_in_order_:
            n += 2
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.entity_)
        for i in range(len(self.entity_)):
            n += self.entity_[i].ByteSizePartial()
        n += 1 * len(self.deferred_)
        for i in range(len(self.deferred_)):
            n += self.lengthString(self.deferred_[i].ByteSizePartial())
        if self.has_in_order_:
            n += 2
        return n

    def Clear(self):
        self.clear_entity()
        self.clear_deferred()
        self.clear_in_order()

    def OutputUnchecked(self, out):
        for i in range(len(self.entity_)):
            out.putVarInt32(11)
            self.entity_[i].OutputUnchecked(out)
            out.putVarInt32(12)
        for i in range(len(self.deferred_)):
            out.putVarInt32(42)
            out.putVarInt32(self.deferred_[i].ByteSize())
            self.deferred_[i].OutputUnchecked(out)
        if self.has_in_order_:
            out.putVarInt32(48)
            out.putBoolean(self.in_order_)

    def OutputPartial(self, out):
        for i in range(len(self.entity_)):
            out.putVarInt32(11)
            self.entity_[i].OutputPartial(out)
            out.putVarInt32(12)
        for i in range(len(self.deferred_)):
            out.putVarInt32(42)
            out.putVarInt32(self.deferred_[i].ByteSizePartial())
            self.deferred_[i].OutputPartial(out)
        if self.has_in_order_:
            out.putVarInt32(48)
            out.putBoolean(self.in_order_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_entity().TryMerge(d)
                continue
            if tt == 42:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_deferred().TryMerge(tmp)
                continue
            if tt == 48:
                self.set_in_order(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.entity_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Entity%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        cnt = 0
        for e in self.deferred_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'deferred%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_in_order_:
            res += prefix + 'in_order: %s\n' % self.DebugFormatBool(self.in_order_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kEntityGroup = 1
    kEntityentity = 2
    kEntitykey = 4
    kEntityversion = 3
    kdeferred = 5
    kin_order = 6
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'Entity', 2: 'entity', 3: 'version', 4: 'key', 5: 'deferred', 6: 'in_order'}, 6)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.STRING, 6: ProtocolBuffer.Encoder.NUMERIC}, 6, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.GetResponse'