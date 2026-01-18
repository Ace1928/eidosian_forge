from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheGetRequest(ProtocolBuffer.ProtocolMessage):
    has_name_space_ = 0
    name_space_ = ''
    has_for_cas_ = 0
    for_cas_ = 0
    has_override_ = 0
    override_ = None

    def __init__(self, contents=None):
        self.key_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def key_size(self):
        return len(self.key_)

    def key_list(self):
        return self.key_

    def key(self, i):
        return self.key_[i]

    def set_key(self, i, x):
        self.key_[i] = x

    def add_key(self, x):
        self.key_.append(x)

    def clear_key(self):
        self.key_ = []

    def name_space(self):
        return self.name_space_

    def set_name_space(self, x):
        self.has_name_space_ = 1
        self.name_space_ = x

    def clear_name_space(self):
        if self.has_name_space_:
            self.has_name_space_ = 0
            self.name_space_ = ''

    def has_name_space(self):
        return self.has_name_space_

    def for_cas(self):
        return self.for_cas_

    def set_for_cas(self, x):
        self.has_for_cas_ = 1
        self.for_cas_ = x

    def clear_for_cas(self):
        if self.has_for_cas_:
            self.has_for_cas_ = 0
            self.for_cas_ = 0

    def has_for_cas(self):
        return self.has_for_cas_

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

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.key_size()):
            self.add_key(x.key(i))
        if x.has_name_space():
            self.set_name_space(x.name_space())
        if x.has_for_cas():
            self.set_for_cas(x.for_cas())
        if x.has_override():
            self.mutable_override().MergeFrom(x.override())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.key_) != len(x.key_):
            return 0
        for e1, e2 in zip(self.key_, x.key_):
            if e1 != e2:
                return 0
        if self.has_name_space_ != x.has_name_space_:
            return 0
        if self.has_name_space_ and self.name_space_ != x.name_space_:
            return 0
        if self.has_for_cas_ != x.has_for_cas_:
            return 0
        if self.has_for_cas_ and self.for_cas_ != x.for_cas_:
            return 0
        if self.has_override_ != x.has_override_:
            return 0
        if self.has_override_ and self.override_ != x.override_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_override_ and (not self.override_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.key_)
        for i in range(len(self.key_)):
            n += self.lengthString(len(self.key_[i]))
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_for_cas_:
            n += 2
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.key_)
        for i in range(len(self.key_)):
            n += self.lengthString(len(self.key_[i]))
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_for_cas_:
            n += 2
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_key()
        self.clear_name_space()
        self.clear_for_cas()
        self.clear_override()

    def OutputUnchecked(self, out):
        for i in range(len(self.key_)):
            out.putVarInt32(10)
            out.putPrefixedString(self.key_[i])
        if self.has_name_space_:
            out.putVarInt32(18)
            out.putPrefixedString(self.name_space_)
        if self.has_for_cas_:
            out.putVarInt32(32)
            out.putBoolean(self.for_cas_)
        if self.has_override_:
            out.putVarInt32(42)
            out.putVarInt32(self.override_.ByteSize())
            self.override_.OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.key_)):
            out.putVarInt32(10)
            out.putPrefixedString(self.key_[i])
        if self.has_name_space_:
            out.putVarInt32(18)
            out.putPrefixedString(self.name_space_)
        if self.has_for_cas_:
            out.putVarInt32(32)
            out.putBoolean(self.for_cas_)
        if self.has_override_:
            out.putVarInt32(42)
            out.putVarInt32(self.override_.ByteSizePartial())
            self.override_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.add_key(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_name_space(d.getPrefixedString())
                continue
            if tt == 32:
                self.set_for_cas(d.getBoolean())
                continue
            if tt == 42:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_override().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.key_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'key%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_name_space_:
            res += prefix + 'name_space: %s\n' % self.DebugFormatString(self.name_space_)
        if self.has_for_cas_:
            res += prefix + 'for_cas: %s\n' % self.DebugFormatBool(self.for_cas_)
        if self.has_override_:
            res += prefix + 'override <\n'
            res += self.override_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kname_space = 2
    kfor_cas = 4
    koverride = 5
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'name_space', 4: 'for_cas', 5: 'override'}, 5)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STRING}, 5, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheGetRequest'