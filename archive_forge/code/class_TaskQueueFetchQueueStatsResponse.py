from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueFetchQueueStatsResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.queuestats_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def queuestats_size(self):
        return len(self.queuestats_)

    def queuestats_list(self):
        return self.queuestats_

    def queuestats(self, i):
        return self.queuestats_[i]

    def mutable_queuestats(self, i):
        return self.queuestats_[i]

    def add_queuestats(self):
        x = TaskQueueFetchQueueStatsResponse_QueueStats()
        self.queuestats_.append(x)
        return x

    def clear_queuestats(self):
        self.queuestats_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.queuestats_size()):
            self.add_queuestats().CopyFrom(x.queuestats(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.queuestats_) != len(x.queuestats_):
            return 0
        for e1, e2 in zip(self.queuestats_, x.queuestats_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.queuestats_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.queuestats_)
        for i in range(len(self.queuestats_)):
            n += self.queuestats_[i].ByteSize()
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.queuestats_)
        for i in range(len(self.queuestats_)):
            n += self.queuestats_[i].ByteSizePartial()
        return n

    def Clear(self):
        self.clear_queuestats()

    def OutputUnchecked(self, out):
        for i in range(len(self.queuestats_)):
            out.putVarInt32(11)
            self.queuestats_[i].OutputUnchecked(out)
            out.putVarInt32(12)

    def OutputPartial(self, out):
        for i in range(len(self.queuestats_)):
            out.putVarInt32(11)
            self.queuestats_[i].OutputPartial(out)
            out.putVarInt32(12)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_queuestats().TryMerge(d)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.queuestats_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'QueueStats%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kQueueStatsGroup = 1
    kQueueStatsnum_tasks = 2
    kQueueStatsoldest_eta_usec = 3
    kQueueStatsscanner_info = 4
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'QueueStats', 2: 'num_tasks', 3: 'oldest_eta_usec', 4: 'scanner_info'}, 4)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.STRING}, 4, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueFetchQueueStatsResponse'