from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueBulkAddResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.taskresult_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def taskresult_size(self):
        return len(self.taskresult_)

    def taskresult_list(self):
        return self.taskresult_

    def taskresult(self, i):
        return self.taskresult_[i]

    def mutable_taskresult(self, i):
        return self.taskresult_[i]

    def add_taskresult(self):
        x = TaskQueueBulkAddResponse_TaskResult()
        self.taskresult_.append(x)
        return x

    def clear_taskresult(self):
        self.taskresult_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.taskresult_size()):
            self.add_taskresult().CopyFrom(x.taskresult(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.taskresult_) != len(x.taskresult_):
            return 0
        for e1, e2 in zip(self.taskresult_, x.taskresult_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.taskresult_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.taskresult_)
        for i in range(len(self.taskresult_)):
            n += self.taskresult_[i].ByteSize()
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.taskresult_)
        for i in range(len(self.taskresult_)):
            n += self.taskresult_[i].ByteSizePartial()
        return n

    def Clear(self):
        self.clear_taskresult()

    def OutputUnchecked(self, out):
        for i in range(len(self.taskresult_)):
            out.putVarInt32(11)
            self.taskresult_[i].OutputUnchecked(out)
            out.putVarInt32(12)

    def OutputPartial(self, out):
        for i in range(len(self.taskresult_)):
            out.putVarInt32(11)
            self.taskresult_[i].OutputPartial(out)
            out.putVarInt32(12)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_taskresult().TryMerge(d)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.taskresult_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'TaskResult%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kTaskResultGroup = 1
    kTaskResultresult = 2
    kTaskResultchosen_task_name = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'TaskResult', 2: 'result', 3: 'chosen_task_name'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.STRING}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueBulkAddResponse'