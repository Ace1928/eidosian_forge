from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryAndOwnTasksResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.task_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def task_size(self):
        return len(self.task_)

    def task_list(self):
        return self.task_

    def task(self, i):
        return self.task_[i]

    def mutable_task(self, i):
        return self.task_[i]

    def add_task(self):
        x = TaskQueueQueryAndOwnTasksResponse_Task()
        self.task_.append(x)
        return x

    def clear_task(self):
        self.task_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.task_size()):
            self.add_task().CopyFrom(x.task(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.task_) != len(x.task_):
            return 0
        for e1, e2 in zip(self.task_, x.task_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.task_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.task_)
        for i in range(len(self.task_)):
            n += self.task_[i].ByteSize()
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.task_)
        for i in range(len(self.task_)):
            n += self.task_[i].ByteSizePartial()
        return n

    def Clear(self):
        self.clear_task()

    def OutputUnchecked(self, out):
        for i in range(len(self.task_)):
            out.putVarInt32(11)
            self.task_[i].OutputUnchecked(out)
            out.putVarInt32(12)

    def OutputPartial(self, out):
        for i in range(len(self.task_)):
            out.putVarInt32(11)
            self.task_[i].OutputPartial(out)
            out.putVarInt32(12)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_task().TryMerge(d)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.task_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Task%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kTaskGroup = 1
    kTasktask_name = 2
    kTasketa_usec = 3
    kTaskretry_count = 4
    kTaskbody = 5
    kTasktag = 6
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'Task', 2: 'task_name', 3: 'eta_usec', 4: 'retry_count', 5: 'body', 6: 'tag'}, 6)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STRING, 6: ProtocolBuffer.Encoder.STRING}, 6, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueQueryAndOwnTasksResponse'