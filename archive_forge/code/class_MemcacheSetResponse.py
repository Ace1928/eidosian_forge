from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheSetResponse(ProtocolBuffer.ProtocolMessage):
    STORED = 1
    NOT_STORED = 2
    ERROR = 3
    EXISTS = 4
    DEADLINE_EXCEEDED = 5
    UNREACHABLE = 6
    OTHER_ERROR = 7
    _SetStatusCode_NAMES = {1: 'STORED', 2: 'NOT_STORED', 3: 'ERROR', 4: 'EXISTS', 5: 'DEADLINE_EXCEEDED', 6: 'UNREACHABLE', 7: 'OTHER_ERROR'}

    def SetStatusCode_Name(cls, x):
        return cls._SetStatusCode_NAMES.get(x, '')
    SetStatusCode_Name = classmethod(SetStatusCode_Name)

    def __init__(self, contents=None):
        self.set_status_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def set_status_size(self):
        return len(self.set_status_)

    def set_status_list(self):
        return self.set_status_

    def set_status(self, i):
        return self.set_status_[i]

    def set_set_status(self, i, x):
        self.set_status_[i] = x

    def add_set_status(self, x):
        self.set_status_.append(x)

    def clear_set_status(self):
        self.set_status_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.set_status_size()):
            self.add_set_status(x.set_status(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.set_status_) != len(x.set_status_):
            return 0
        for e1, e2 in zip(self.set_status_, x.set_status_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.set_status_)
        for i in range(len(self.set_status_)):
            n += self.lengthVarInt64(self.set_status_[i])
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.set_status_)
        for i in range(len(self.set_status_)):
            n += self.lengthVarInt64(self.set_status_[i])
        return n

    def Clear(self):
        self.clear_set_status()

    def OutputUnchecked(self, out):
        for i in range(len(self.set_status_)):
            out.putVarInt32(8)
            out.putVarInt32(self.set_status_[i])

    def OutputPartial(self, out):
        for i in range(len(self.set_status_)):
            out.putVarInt32(8)
            out.putVarInt32(self.set_status_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.add_set_status(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.set_status_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'set_status%s: %s\n' % (elm, self.DebugFormatInt32(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kset_status = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'set_status'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheSetResponse'