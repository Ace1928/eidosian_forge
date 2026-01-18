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
class Cost(ProtocolBuffer.ProtocolMessage):
    has_index_writes_ = 0
    index_writes_ = 0
    has_index_write_bytes_ = 0
    index_write_bytes_ = 0
    has_entity_writes_ = 0
    entity_writes_ = 0
    has_entity_write_bytes_ = 0
    entity_write_bytes_ = 0
    has_commitcost_ = 0
    commitcost_ = None
    has_approximate_storage_delta_ = 0
    approximate_storage_delta_ = 0
    has_id_sequence_updates_ = 0
    id_sequence_updates_ = 0

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def index_writes(self):
        return self.index_writes_

    def set_index_writes(self, x):
        self.has_index_writes_ = 1
        self.index_writes_ = x

    def clear_index_writes(self):
        if self.has_index_writes_:
            self.has_index_writes_ = 0
            self.index_writes_ = 0

    def has_index_writes(self):
        return self.has_index_writes_

    def index_write_bytes(self):
        return self.index_write_bytes_

    def set_index_write_bytes(self, x):
        self.has_index_write_bytes_ = 1
        self.index_write_bytes_ = x

    def clear_index_write_bytes(self):
        if self.has_index_write_bytes_:
            self.has_index_write_bytes_ = 0
            self.index_write_bytes_ = 0

    def has_index_write_bytes(self):
        return self.has_index_write_bytes_

    def entity_writes(self):
        return self.entity_writes_

    def set_entity_writes(self, x):
        self.has_entity_writes_ = 1
        self.entity_writes_ = x

    def clear_entity_writes(self):
        if self.has_entity_writes_:
            self.has_entity_writes_ = 0
            self.entity_writes_ = 0

    def has_entity_writes(self):
        return self.has_entity_writes_

    def entity_write_bytes(self):
        return self.entity_write_bytes_

    def set_entity_write_bytes(self, x):
        self.has_entity_write_bytes_ = 1
        self.entity_write_bytes_ = x

    def clear_entity_write_bytes(self):
        if self.has_entity_write_bytes_:
            self.has_entity_write_bytes_ = 0
            self.entity_write_bytes_ = 0

    def has_entity_write_bytes(self):
        return self.has_entity_write_bytes_

    def commitcost(self):
        if self.commitcost_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.commitcost_ is None:
                    self.commitcost_ = Cost_CommitCost()
            finally:
                self.lazy_init_lock_.release()
        return self.commitcost_

    def mutable_commitcost(self):
        self.has_commitcost_ = 1
        return self.commitcost()

    def clear_commitcost(self):
        if self.has_commitcost_:
            self.has_commitcost_ = 0
            if self.commitcost_ is not None:
                self.commitcost_.Clear()

    def has_commitcost(self):
        return self.has_commitcost_

    def approximate_storage_delta(self):
        return self.approximate_storage_delta_

    def set_approximate_storage_delta(self, x):
        self.has_approximate_storage_delta_ = 1
        self.approximate_storage_delta_ = x

    def clear_approximate_storage_delta(self):
        if self.has_approximate_storage_delta_:
            self.has_approximate_storage_delta_ = 0
            self.approximate_storage_delta_ = 0

    def has_approximate_storage_delta(self):
        return self.has_approximate_storage_delta_

    def id_sequence_updates(self):
        return self.id_sequence_updates_

    def set_id_sequence_updates(self, x):
        self.has_id_sequence_updates_ = 1
        self.id_sequence_updates_ = x

    def clear_id_sequence_updates(self):
        if self.has_id_sequence_updates_:
            self.has_id_sequence_updates_ = 0
            self.id_sequence_updates_ = 0

    def has_id_sequence_updates(self):
        return self.has_id_sequence_updates_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_writes():
            self.set_index_writes(x.index_writes())
        if x.has_index_write_bytes():
            self.set_index_write_bytes(x.index_write_bytes())
        if x.has_entity_writes():
            self.set_entity_writes(x.entity_writes())
        if x.has_entity_write_bytes():
            self.set_entity_write_bytes(x.entity_write_bytes())
        if x.has_commitcost():
            self.mutable_commitcost().MergeFrom(x.commitcost())
        if x.has_approximate_storage_delta():
            self.set_approximate_storage_delta(x.approximate_storage_delta())
        if x.has_id_sequence_updates():
            self.set_id_sequence_updates(x.id_sequence_updates())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_writes_ != x.has_index_writes_:
            return 0
        if self.has_index_writes_ and self.index_writes_ != x.index_writes_:
            return 0
        if self.has_index_write_bytes_ != x.has_index_write_bytes_:
            return 0
        if self.has_index_write_bytes_ and self.index_write_bytes_ != x.index_write_bytes_:
            return 0
        if self.has_entity_writes_ != x.has_entity_writes_:
            return 0
        if self.has_entity_writes_ and self.entity_writes_ != x.entity_writes_:
            return 0
        if self.has_entity_write_bytes_ != x.has_entity_write_bytes_:
            return 0
        if self.has_entity_write_bytes_ and self.entity_write_bytes_ != x.entity_write_bytes_:
            return 0
        if self.has_commitcost_ != x.has_commitcost_:
            return 0
        if self.has_commitcost_ and self.commitcost_ != x.commitcost_:
            return 0
        if self.has_approximate_storage_delta_ != x.has_approximate_storage_delta_:
            return 0
        if self.has_approximate_storage_delta_ and self.approximate_storage_delta_ != x.approximate_storage_delta_:
            return 0
        if self.has_id_sequence_updates_ != x.has_id_sequence_updates_:
            return 0
        if self.has_id_sequence_updates_ and self.id_sequence_updates_ != x.id_sequence_updates_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_commitcost_ and (not self.commitcost_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_index_writes_:
            n += 1 + self.lengthVarInt64(self.index_writes_)
        if self.has_index_write_bytes_:
            n += 1 + self.lengthVarInt64(self.index_write_bytes_)
        if self.has_entity_writes_:
            n += 1 + self.lengthVarInt64(self.entity_writes_)
        if self.has_entity_write_bytes_:
            n += 1 + self.lengthVarInt64(self.entity_write_bytes_)
        if self.has_commitcost_:
            n += 2 + self.commitcost_.ByteSize()
        if self.has_approximate_storage_delta_:
            n += 1 + self.lengthVarInt64(self.approximate_storage_delta_)
        if self.has_id_sequence_updates_:
            n += 1 + self.lengthVarInt64(self.id_sequence_updates_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_index_writes_:
            n += 1 + self.lengthVarInt64(self.index_writes_)
        if self.has_index_write_bytes_:
            n += 1 + self.lengthVarInt64(self.index_write_bytes_)
        if self.has_entity_writes_:
            n += 1 + self.lengthVarInt64(self.entity_writes_)
        if self.has_entity_write_bytes_:
            n += 1 + self.lengthVarInt64(self.entity_write_bytes_)
        if self.has_commitcost_:
            n += 2 + self.commitcost_.ByteSizePartial()
        if self.has_approximate_storage_delta_:
            n += 1 + self.lengthVarInt64(self.approximate_storage_delta_)
        if self.has_id_sequence_updates_:
            n += 1 + self.lengthVarInt64(self.id_sequence_updates_)
        return n

    def Clear(self):
        self.clear_index_writes()
        self.clear_index_write_bytes()
        self.clear_entity_writes()
        self.clear_entity_write_bytes()
        self.clear_commitcost()
        self.clear_approximate_storage_delta()
        self.clear_id_sequence_updates()

    def OutputUnchecked(self, out):
        if self.has_index_writes_:
            out.putVarInt32(8)
            out.putVarInt32(self.index_writes_)
        if self.has_index_write_bytes_:
            out.putVarInt32(16)
            out.putVarInt32(self.index_write_bytes_)
        if self.has_entity_writes_:
            out.putVarInt32(24)
            out.putVarInt32(self.entity_writes_)
        if self.has_entity_write_bytes_:
            out.putVarInt32(32)
            out.putVarInt32(self.entity_write_bytes_)
        if self.has_commitcost_:
            out.putVarInt32(43)
            self.commitcost_.OutputUnchecked(out)
            out.putVarInt32(44)
        if self.has_approximate_storage_delta_:
            out.putVarInt32(64)
            out.putVarInt32(self.approximate_storage_delta_)
        if self.has_id_sequence_updates_:
            out.putVarInt32(72)
            out.putVarInt32(self.id_sequence_updates_)

    def OutputPartial(self, out):
        if self.has_index_writes_:
            out.putVarInt32(8)
            out.putVarInt32(self.index_writes_)
        if self.has_index_write_bytes_:
            out.putVarInt32(16)
            out.putVarInt32(self.index_write_bytes_)
        if self.has_entity_writes_:
            out.putVarInt32(24)
            out.putVarInt32(self.entity_writes_)
        if self.has_entity_write_bytes_:
            out.putVarInt32(32)
            out.putVarInt32(self.entity_write_bytes_)
        if self.has_commitcost_:
            out.putVarInt32(43)
            self.commitcost_.OutputPartial(out)
            out.putVarInt32(44)
        if self.has_approximate_storage_delta_:
            out.putVarInt32(64)
            out.putVarInt32(self.approximate_storage_delta_)
        if self.has_id_sequence_updates_:
            out.putVarInt32(72)
            out.putVarInt32(self.id_sequence_updates_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_index_writes(d.getVarInt32())
                continue
            if tt == 16:
                self.set_index_write_bytes(d.getVarInt32())
                continue
            if tt == 24:
                self.set_entity_writes(d.getVarInt32())
                continue
            if tt == 32:
                self.set_entity_write_bytes(d.getVarInt32())
                continue
            if tt == 43:
                self.mutable_commitcost().TryMerge(d)
                continue
            if tt == 64:
                self.set_approximate_storage_delta(d.getVarInt32())
                continue
            if tt == 72:
                self.set_id_sequence_updates(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_writes_:
            res += prefix + 'index_writes: %s\n' % self.DebugFormatInt32(self.index_writes_)
        if self.has_index_write_bytes_:
            res += prefix + 'index_write_bytes: %s\n' % self.DebugFormatInt32(self.index_write_bytes_)
        if self.has_entity_writes_:
            res += prefix + 'entity_writes: %s\n' % self.DebugFormatInt32(self.entity_writes_)
        if self.has_entity_write_bytes_:
            res += prefix + 'entity_write_bytes: %s\n' % self.DebugFormatInt32(self.entity_write_bytes_)
        if self.has_commitcost_:
            res += prefix + 'CommitCost {\n'
            res += self.commitcost_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_approximate_storage_delta_:
            res += prefix + 'approximate_storage_delta: %s\n' % self.DebugFormatInt32(self.approximate_storage_delta_)
        if self.has_id_sequence_updates_:
            res += prefix + 'id_sequence_updates: %s\n' % self.DebugFormatInt32(self.id_sequence_updates_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex_writes = 1
    kindex_write_bytes = 2
    kentity_writes = 3
    kentity_write_bytes = 4
    kCommitCostGroup = 5
    kCommitCostrequested_entity_puts = 6
    kCommitCostrequested_entity_deletes = 7
    kapproximate_storage_delta = 8
    kid_sequence_updates = 9
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index_writes', 2: 'index_write_bytes', 3: 'entity_writes', 4: 'entity_write_bytes', 5: 'CommitCost', 6: 'requested_entity_puts', 7: 'requested_entity_deletes', 8: 'approximate_storage_delta', 9: 'id_sequence_updates'}, 9)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STARTGROUP, 6: ProtocolBuffer.Encoder.NUMERIC, 7: ProtocolBuffer.Encoder.NUMERIC, 8: ProtocolBuffer.Encoder.NUMERIC, 9: ProtocolBuffer.Encoder.NUMERIC}, 9, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.Cost'