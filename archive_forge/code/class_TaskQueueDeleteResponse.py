from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueDeleteResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.result_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def result_size(self):
        return len(self.result_)

    def result_list(self):
        return self.result_

    def result(self, i):
        return self.result_[i]

    def set_result(self, i, x):
        self.result_[i] = x

    def add_result(self, x):
        self.result_.append(x)

    def clear_result(self):
        self.result_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.result_size()):
            self.add_result(x.result(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.result_) != len(x.result_):
            return 0
        for e1, e2 in zip(self.result_, x.result_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.result_)
        for i in range(len(self.result_)):
            n += self.lengthVarInt64(self.result_[i])
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.result_)
        for i in range(len(self.result_)):
            n += self.lengthVarInt64(self.result_[i])
        return n

    def Clear(self):
        self.clear_result()

    def OutputUnchecked(self, out):
        for i in range(len(self.result_)):
            out.putVarInt32(24)
            out.putVarInt32(self.result_[i])

    def OutputPartial(self, out):
        for i in range(len(self.result_)):
            out.putVarInt32(24)
            out.putVarInt32(self.result_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 24:
                self.add_result(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.result_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'result%s: %s\n' % (elm, self.DebugFormatInt32(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kresult = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 3: 'result'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueDeleteResponse'