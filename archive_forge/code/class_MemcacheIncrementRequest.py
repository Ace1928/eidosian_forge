from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheIncrementRequest(ProtocolBuffer.ProtocolMessage):
    INCREMENT = 1
    DECREMENT = 2
    _Direction_NAMES = {1: 'INCREMENT', 2: 'DECREMENT'}

    def Direction_Name(cls, x):
        return cls._Direction_NAMES.get(x, '')
    Direction_Name = classmethod(Direction_Name)
    has_key_ = 0
    key_ = ''
    has_name_space_ = 0
    name_space_ = ''
    has_delta_ = 0
    delta_ = 1
    has_direction_ = 0
    direction_ = 1
    has_initial_value_ = 0
    initial_value_ = 0
    has_initial_flags_ = 0
    initial_flags_ = 0
    has_override_ = 0
    override_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def key(self):
        return self.key_

    def set_key(self, x):
        self.has_key_ = 1
        self.key_ = x

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            self.key_ = ''

    def has_key(self):
        return self.has_key_

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

    def delta(self):
        return self.delta_

    def set_delta(self, x):
        self.has_delta_ = 1
        self.delta_ = x

    def clear_delta(self):
        if self.has_delta_:
            self.has_delta_ = 0
            self.delta_ = 1

    def has_delta(self):
        return self.has_delta_

    def direction(self):
        return self.direction_

    def set_direction(self, x):
        self.has_direction_ = 1
        self.direction_ = x

    def clear_direction(self):
        if self.has_direction_:
            self.has_direction_ = 0
            self.direction_ = 1

    def has_direction(self):
        return self.has_direction_

    def initial_value(self):
        return self.initial_value_

    def set_initial_value(self, x):
        self.has_initial_value_ = 1
        self.initial_value_ = x

    def clear_initial_value(self):
        if self.has_initial_value_:
            self.has_initial_value_ = 0
            self.initial_value_ = 0

    def has_initial_value(self):
        return self.has_initial_value_

    def initial_flags(self):
        return self.initial_flags_

    def set_initial_flags(self, x):
        self.has_initial_flags_ = 1
        self.initial_flags_ = x

    def clear_initial_flags(self):
        if self.has_initial_flags_:
            self.has_initial_flags_ = 0
            self.initial_flags_ = 0

    def has_initial_flags(self):
        return self.has_initial_flags_

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
        if x.has_key():
            self.set_key(x.key())
        if x.has_name_space():
            self.set_name_space(x.name_space())
        if x.has_delta():
            self.set_delta(x.delta())
        if x.has_direction():
            self.set_direction(x.direction())
        if x.has_initial_value():
            self.set_initial_value(x.initial_value())
        if x.has_initial_flags():
            self.set_initial_flags(x.initial_flags())
        if x.has_override():
            self.mutable_override().MergeFrom(x.override())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_name_space_ != x.has_name_space_:
            return 0
        if self.has_name_space_ and self.name_space_ != x.name_space_:
            return 0
        if self.has_delta_ != x.has_delta_:
            return 0
        if self.has_delta_ and self.delta_ != x.delta_:
            return 0
        if self.has_direction_ != x.has_direction_:
            return 0
        if self.has_direction_ and self.direction_ != x.direction_:
            return 0
        if self.has_initial_value_ != x.has_initial_value_:
            return 0
        if self.has_initial_value_ and self.initial_value_ != x.initial_value_:
            return 0
        if self.has_initial_flags_ != x.has_initial_flags_:
            return 0
        if self.has_initial_flags_ and self.initial_flags_ != x.initial_flags_:
            return 0
        if self.has_override_ != x.has_override_:
            return 0
        if self.has_override_ and self.override_ != x.override_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        if self.has_override_ and (not self.override_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_delta_:
            n += 1 + self.lengthVarInt64(self.delta_)
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        if self.has_initial_value_:
            n += 1 + self.lengthVarInt64(self.initial_value_)
        if self.has_initial_flags_:
            n += 5
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSize())
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_delta_:
            n += 1 + self.lengthVarInt64(self.delta_)
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        if self.has_initial_value_:
            n += 1 + self.lengthVarInt64(self.initial_value_)
        if self.has_initial_flags_:
            n += 5
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_key()
        self.clear_name_space()
        self.clear_delta()
        self.clear_direction()
        self.clear_initial_value()
        self.clear_initial_flags()
        self.clear_override()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.key_)
        if self.has_delta_:
            out.putVarInt32(16)
            out.putVarUint64(self.delta_)
        if self.has_direction_:
            out.putVarInt32(24)
            out.putVarInt32(self.direction_)
        if self.has_name_space_:
            out.putVarInt32(34)
            out.putPrefixedString(self.name_space_)
        if self.has_initial_value_:
            out.putVarInt32(40)
            out.putVarUint64(self.initial_value_)
        if self.has_initial_flags_:
            out.putVarInt32(53)
            out.put32(self.initial_flags_)
        if self.has_override_:
            out.putVarInt32(58)
            out.putVarInt32(self.override_.ByteSize())
            self.override_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(10)
            out.putPrefixedString(self.key_)
        if self.has_delta_:
            out.putVarInt32(16)
            out.putVarUint64(self.delta_)
        if self.has_direction_:
            out.putVarInt32(24)
            out.putVarInt32(self.direction_)
        if self.has_name_space_:
            out.putVarInt32(34)
            out.putPrefixedString(self.name_space_)
        if self.has_initial_value_:
            out.putVarInt32(40)
            out.putVarUint64(self.initial_value_)
        if self.has_initial_flags_:
            out.putVarInt32(53)
            out.put32(self.initial_flags_)
        if self.has_override_:
            out.putVarInt32(58)
            out.putVarInt32(self.override_.ByteSizePartial())
            self.override_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 16:
                self.set_delta(d.getVarUint64())
                continue
            if tt == 24:
                self.set_direction(d.getVarInt32())
                continue
            if tt == 34:
                self.set_name_space(d.getPrefixedString())
                continue
            if tt == 40:
                self.set_initial_value(d.getVarUint64())
                continue
            if tt == 53:
                self.set_initial_flags(d.get32())
                continue
            if tt == 58:
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
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_name_space_:
            res += prefix + 'name_space: %s\n' % self.DebugFormatString(self.name_space_)
        if self.has_delta_:
            res += prefix + 'delta: %s\n' % self.DebugFormatInt64(self.delta_)
        if self.has_direction_:
            res += prefix + 'direction: %s\n' % self.DebugFormatInt32(self.direction_)
        if self.has_initial_value_:
            res += prefix + 'initial_value: %s\n' % self.DebugFormatInt64(self.initial_value_)
        if self.has_initial_flags_:
            res += prefix + 'initial_flags: %s\n' % self.DebugFormatFixed32(self.initial_flags_)
        if self.has_override_:
            res += prefix + 'override <\n'
            res += self.override_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kname_space = 4
    kdelta = 2
    kdirection = 3
    kinitial_value = 5
    kinitial_flags = 6
    koverride = 7
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'delta', 3: 'direction', 4: 'name_space', 5: 'initial_value', 6: 'initial_flags', 7: 'override'}, 7)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.FLOAT, 7: ProtocolBuffer.Encoder.STRING}, 7, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheIncrementRequest'