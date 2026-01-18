from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheDeleteResponse(ProtocolBuffer.ProtocolMessage):
    DELETED = 1
    NOT_FOUND = 2
    DEADLINE_EXCEEDED = 3
    UNREACHABLE = 4
    OTHER_ERROR = 5
    _DeleteStatusCode_NAMES = {1: 'DELETED', 2: 'NOT_FOUND', 3: 'DEADLINE_EXCEEDED', 4: 'UNREACHABLE', 5: 'OTHER_ERROR'}

    def DeleteStatusCode_Name(cls, x):
        return cls._DeleteStatusCode_NAMES.get(x, '')
    DeleteStatusCode_Name = classmethod(DeleteStatusCode_Name)

    def __init__(self, contents=None):
        self.delete_status_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def delete_status_size(self):
        return len(self.delete_status_)

    def delete_status_list(self):
        return self.delete_status_

    def delete_status(self, i):
        return self.delete_status_[i]

    def set_delete_status(self, i, x):
        self.delete_status_[i] = x

    def add_delete_status(self, x):
        self.delete_status_.append(x)

    def clear_delete_status(self):
        self.delete_status_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.delete_status_size()):
            self.add_delete_status(x.delete_status(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.delete_status_) != len(x.delete_status_):
            return 0
        for e1, e2 in zip(self.delete_status_, x.delete_status_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.delete_status_)
        for i in range(len(self.delete_status_)):
            n += self.lengthVarInt64(self.delete_status_[i])
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.delete_status_)
        for i in range(len(self.delete_status_)):
            n += self.lengthVarInt64(self.delete_status_[i])
        return n

    def Clear(self):
        self.clear_delete_status()

    def OutputUnchecked(self, out):
        for i in range(len(self.delete_status_)):
            out.putVarInt32(8)
            out.putVarInt32(self.delete_status_[i])

    def OutputPartial(self, out):
        for i in range(len(self.delete_status_)):
            out.putVarInt32(8)
            out.putVarInt32(self.delete_status_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.add_delete_status(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.delete_status_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'delete_status%s: %s\n' % (elm, self.DebugFormatInt32(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kdelete_status = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'delete_status'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheDeleteResponse'