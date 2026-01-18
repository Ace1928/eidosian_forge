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
class TouchRequest(ProtocolBuffer.ProtocolMessage):
    has_force_ = 0
    force_ = 0

    def __init__(self, contents=None):
        self.key_ = []
        self.composite_index_ = []
        self.snapshot_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def key_size(self):
        return len(self.key_)

    def key_list(self):
        return self.key_

    def key(self, i):
        return self.key_[i]

    def mutable_key(self, i):
        return self.key_[i]

    def add_key(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
        self.key_.append(x)
        return x

    def clear_key(self):
        self.key_ = []

    def composite_index_size(self):
        return len(self.composite_index_)

    def composite_index_list(self):
        return self.composite_index_

    def composite_index(self, i):
        return self.composite_index_[i]

    def mutable_composite_index(self, i):
        return self.composite_index_[i]

    def add_composite_index(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.CompositeIndex()
        self.composite_index_.append(x)
        return x

    def clear_composite_index(self):
        self.composite_index_ = []

    def force(self):
        return self.force_

    def set_force(self, x):
        self.has_force_ = 1
        self.force_ = x

    def clear_force(self):
        if self.has_force_:
            self.has_force_ = 0
            self.force_ = 0

    def has_force(self):
        return self.has_force_

    def snapshot_size(self):
        return len(self.snapshot_)

    def snapshot_list(self):
        return self.snapshot_

    def snapshot(self, i):
        return self.snapshot_[i]

    def mutable_snapshot(self, i):
        return self.snapshot_[i]

    def add_snapshot(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb.Snapshot()
        self.snapshot_.append(x)
        return x

    def clear_snapshot(self):
        self.snapshot_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.key_size()):
            self.add_key().CopyFrom(x.key(i))
        for i in range(x.composite_index_size()):
            self.add_composite_index().CopyFrom(x.composite_index(i))
        if x.has_force():
            self.set_force(x.force())
        for i in range(x.snapshot_size()):
            self.add_snapshot().CopyFrom(x.snapshot(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.key_) != len(x.key_):
            return 0
        for e1, e2 in zip(self.key_, x.key_):
            if e1 != e2:
                return 0
        if len(self.composite_index_) != len(x.composite_index_):
            return 0
        for e1, e2 in zip(self.composite_index_, x.composite_index_):
            if e1 != e2:
                return 0
        if self.has_force_ != x.has_force_:
            return 0
        if self.has_force_ and self.force_ != x.force_:
            return 0
        if len(self.snapshot_) != len(x.snapshot_):
            return 0
        for e1, e2 in zip(self.snapshot_, x.snapshot_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.key_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.composite_index_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.snapshot_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.key_)
        for i in range(len(self.key_)):
            n += self.lengthString(self.key_[i].ByteSize())
        n += 1 * len(self.composite_index_)
        for i in range(len(self.composite_index_)):
            n += self.lengthString(self.composite_index_[i].ByteSize())
        if self.has_force_:
            n += 2
        n += 1 * len(self.snapshot_)
        for i in range(len(self.snapshot_)):
            n += self.lengthString(self.snapshot_[i].ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.key_)
        for i in range(len(self.key_)):
            n += self.lengthString(self.key_[i].ByteSizePartial())
        n += 1 * len(self.composite_index_)
        for i in range(len(self.composite_index_)):
            n += self.lengthString(self.composite_index_[i].ByteSizePartial())
        if self.has_force_:
            n += 2
        n += 1 * len(self.snapshot_)
        for i in range(len(self.snapshot_)):
            n += self.lengthString(self.snapshot_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_key()
        self.clear_composite_index()
        self.clear_force()
        self.clear_snapshot()

    def OutputUnchecked(self, out):
        for i in range(len(self.key_)):
            out.putVarInt32(10)
            out.putVarInt32(self.key_[i].ByteSize())
            self.key_[i].OutputUnchecked(out)
        for i in range(len(self.composite_index_)):
            out.putVarInt32(18)
            out.putVarInt32(self.composite_index_[i].ByteSize())
            self.composite_index_[i].OutputUnchecked(out)
        if self.has_force_:
            out.putVarInt32(24)
            out.putBoolean(self.force_)
        for i in range(len(self.snapshot_)):
            out.putVarInt32(74)
            out.putVarInt32(self.snapshot_[i].ByteSize())
            self.snapshot_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.key_)):
            out.putVarInt32(10)
            out.putVarInt32(self.key_[i].ByteSizePartial())
            self.key_[i].OutputPartial(out)
        for i in range(len(self.composite_index_)):
            out.putVarInt32(18)
            out.putVarInt32(self.composite_index_[i].ByteSizePartial())
            self.composite_index_[i].OutputPartial(out)
        if self.has_force_:
            out.putVarInt32(24)
            out.putBoolean(self.force_)
        for i in range(len(self.snapshot_)):
            out.putVarInt32(74)
            out.putVarInt32(self.snapshot_[i].ByteSizePartial())
            self.snapshot_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_key().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_composite_index().TryMerge(tmp)
                continue
            if tt == 24:
                self.set_force(d.getBoolean())
                continue
            if tt == 74:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_snapshot().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.key_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'key%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.composite_index_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'composite_index%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_force_:
            res += prefix + 'force: %s\n' % self.DebugFormatBool(self.force_)
        cnt = 0
        for e in self.snapshot_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'snapshot%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kcomposite_index = 2
    kforce = 3
    ksnapshot = 9
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'composite_index', 3: 'force', 9: 'snapshot'}, 9)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 9: ProtocolBuffer.Encoder.STRING}, 9, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.TouchRequest'