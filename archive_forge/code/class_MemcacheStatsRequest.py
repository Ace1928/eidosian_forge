from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheStatsRequest(ProtocolBuffer.ProtocolMessage):
    has_override_ = 0
    override_ = None
    has_max_hotkey_count_ = 0
    max_hotkey_count_ = 0

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def override(self):
        if self.override_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.override_ is None:
                    self.override_ = AppOverride()
            finally:
                self.lazy_init_lock_.release()
        return self.override_

    def mutable_override(self):
        self.has_override_ = 1
        return self.override()

    def clear_override(self):
        if self.has_override_:
            self.has_override_ = 0
            if self.override_ is not None:
                self.override_.Clear()

    def has_override(self):
        return self.has_override_

    def max_hotkey_count(self):
        return self.max_hotkey_count_

    def set_max_hotkey_count(self, x):
        self.has_max_hotkey_count_ = 1
        self.max_hotkey_count_ = x

    def clear_max_hotkey_count(self):
        if self.has_max_hotkey_count_:
            self.has_max_hotkey_count_ = 0
            self.max_hotkey_count_ = 0

    def has_max_hotkey_count(self):
        return self.has_max_hotkey_count_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_override():
            self.mutable_override().MergeFrom(x.override())
        if x.has_max_hotkey_count():
            self.set_max_hotkey_count(x.max_hotkey_count())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_override_ != x.has_override_:
            return 0
        if self.has_override_ and self.override_ != x.override_:
            return 0
        if self.has_max_hotkey_count_ != x.has_max_hotkey_count_:
            return 0
        if self.has_max_hotkey_count_ and self.max_hotkey_count_ != x.max_hotkey_count_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_override_ and (not self.override_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSize())
        if self.has_max_hotkey_count_:
            n += 1 + self.lengthVarInt64(self.max_hotkey_count_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSizePartial())
        if self.has_max_hotkey_count_:
            n += 1 + self.lengthVarInt64(self.max_hotkey_count_)
        return n

    def Clear(self):
        self.clear_override()
        self.clear_max_hotkey_count()

    def OutputUnchecked(self, out):
        if self.has_override_:
            out.putVarInt32(10)
            out.putVarInt32(self.override_.ByteSize())
            self.override_.OutputUnchecked(out)
        if self.has_max_hotkey_count_:
            out.putVarInt32(16)
            out.putVarInt32(self.max_hotkey_count_)

    def OutputPartial(self, out):
        if self.has_override_:
            out.putVarInt32(10)
            out.putVarInt32(self.override_.ByteSizePartial())
            self.override_.OutputPartial(out)
        if self.has_max_hotkey_count_:
            out.putVarInt32(16)
            out.putVarInt32(self.max_hotkey_count_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_override().TryMerge(tmp)
                continue
            if tt == 16:
                self.set_max_hotkey_count(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_override_:
            res += prefix + 'override <\n'
            res += self.override_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_max_hotkey_count_:
            res += prefix + 'max_hotkey_count: %s\n' % self.DebugFormatInt32(self.max_hotkey_count_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    koverride = 1
    kmax_hotkey_count = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'override', 2: 'max_hotkey_count'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheStatsRequest'