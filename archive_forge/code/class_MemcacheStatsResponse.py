from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheStatsResponse(ProtocolBuffer.ProtocolMessage):
    has_stats_ = 0
    stats_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def stats(self):
        if self.stats_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.stats_ is None:
                    self.stats_ = MergedNamespaceStats()
            finally:
                self.lazy_init_lock_.release()
        return self.stats_

    def mutable_stats(self):
        self.has_stats_ = 1
        return self.stats()

    def clear_stats(self):
        if self.has_stats_:
            self.has_stats_ = 0
            if self.stats_ is not None:
                self.stats_.Clear()

    def has_stats(self):
        return self.has_stats_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_stats():
            self.mutable_stats().MergeFrom(x.stats())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_stats_ != x.has_stats_:
            return 0
        if self.has_stats_ and self.stats_ != x.stats_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_stats_ and (not self.stats_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_stats_:
            n += 1 + self.lengthString(self.stats_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_stats_:
            n += 1 + self.lengthString(self.stats_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_stats()

    def OutputUnchecked(self, out):
        if self.has_stats_:
            out.putVarInt32(10)
            out.putVarInt32(self.stats_.ByteSize())
            self.stats_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_stats_:
            out.putVarInt32(10)
            out.putVarInt32(self.stats_.ByteSizePartial())
            self.stats_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_stats().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_stats_:
            res += prefix + 'stats <\n'
            res += self.stats_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kstats = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'stats'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheStatsResponse'