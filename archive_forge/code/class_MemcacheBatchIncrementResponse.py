from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheBatchIncrementResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.item_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def item_size(self):
        return len(self.item_)

    def item_list(self):
        return self.item_

    def item(self, i):
        return self.item_[i]

    def mutable_item(self, i):
        return self.item_[i]

    def add_item(self):
        x = MemcacheIncrementResponse()
        self.item_.append(x)
        return x

    def clear_item(self):
        self.item_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.item_size()):
            self.add_item().CopyFrom(x.item(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.item_) != len(x.item_):
            return 0
        for e1, e2 in zip(self.item_, x.item_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.item_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.lengthString(self.item_[i].ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.lengthString(self.item_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_item()

    def OutputUnchecked(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(10)
            out.putVarInt32(self.item_[i].ByteSize())
            self.item_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(10)
            out.putVarInt32(self.item_[i].ByteSizePartial())
            self.item_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_item().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.item_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'item%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kitem = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'item'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheBatchIncrementResponse'