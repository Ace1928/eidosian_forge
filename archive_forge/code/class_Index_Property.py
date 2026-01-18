from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class Index_Property(ProtocolBuffer.ProtocolMessage):
    DIRECTION_UNSPECIFIED = 0
    ASCENDING = 1
    DESCENDING = 2
    _Direction_NAMES = {0: 'DIRECTION_UNSPECIFIED', 1: 'ASCENDING', 2: 'DESCENDING'}

    def Direction_Name(cls, x):
        return cls._Direction_NAMES.get(x, '')
    Direction_Name = classmethod(Direction_Name)
    MODE_UNSPECIFIED = 0
    GEOSPATIAL = 3
    ARRAY_CONTAINS = 4
    _Mode_NAMES = {0: 'MODE_UNSPECIFIED', 3: 'GEOSPATIAL', 4: 'ARRAY_CONTAINS'}

    def Mode_Name(cls, x):
        return cls._Mode_NAMES.get(x, '')
    Mode_Name = classmethod(Mode_Name)
    has_name_ = 0
    name_ = ''
    has_direction_ = 0
    direction_ = 0
    has_mode_ = 0
    mode_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def name(self):
        return self.name_

    def set_name(self, x):
        self.has_name_ = 1
        self.name_ = x

    def clear_name(self):
        if self.has_name_:
            self.has_name_ = 0
            self.name_ = ''

    def has_name(self):
        return self.has_name_

    def direction(self):
        return self.direction_

    def set_direction(self, x):
        self.has_direction_ = 1
        self.direction_ = x

    def clear_direction(self):
        if self.has_direction_:
            self.has_direction_ = 0
            self.direction_ = 0

    def has_direction(self):
        return self.has_direction_

    def mode(self):
        return self.mode_

    def set_mode(self, x):
        self.has_mode_ = 1
        self.mode_ = x

    def clear_mode(self):
        if self.has_mode_:
            self.has_mode_ = 0
            self.mode_ = 0

    def has_mode(self):
        return self.has_mode_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_name():
            self.set_name(x.name())
        if x.has_direction():
            self.set_direction(x.direction())
        if x.has_mode():
            self.set_mode(x.mode())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_name_ != x.has_name_:
            return 0
        if self.has_name_ and self.name_ != x.name_:
            return 0
        if self.has_direction_ != x.has_direction_:
            return 0
        if self.has_direction_ and self.direction_ != x.direction_:
            return 0
        if self.has_mode_ != x.has_mode_:
            return 0
        if self.has_mode_ and self.mode_ != x.mode_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: name not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.name_))
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        if self.has_mode_:
            n += 1 + self.lengthVarInt64(self.mode_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_name_:
            n += 1
            n += self.lengthString(len(self.name_))
        if self.has_direction_:
            n += 1 + self.lengthVarInt64(self.direction_)
        if self.has_mode_:
            n += 1 + self.lengthVarInt64(self.mode_)
        return n

    def Clear(self):
        self.clear_name()
        self.clear_direction()
        self.clear_mode()

    def OutputUnchecked(self, out):
        out.putVarInt32(26)
        out.putPrefixedString(self.name_)
        if self.has_direction_:
            out.putVarInt32(32)
            out.putVarInt32(self.direction_)
        if self.has_mode_:
            out.putVarInt32(48)
            out.putVarInt32(self.mode_)

    def OutputPartial(self, out):
        if self.has_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.name_)
        if self.has_direction_:
            out.putVarInt32(32)
            out.putVarInt32(self.direction_)
        if self.has_mode_:
            out.putVarInt32(48)
            out.putVarInt32(self.mode_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 20:
                break
            if tt == 26:
                self.set_name(d.getPrefixedString())
                continue
            if tt == 32:
                self.set_direction(d.getVarInt32())
                continue
            if tt == 48:
                self.set_mode(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_name_:
            res += prefix + 'name: %s\n' % self.DebugFormatString(self.name_)
        if self.has_direction_:
            res += prefix + 'direction: %s\n' % self.DebugFormatInt32(self.direction_)
        if self.has_mode_:
            res += prefix + 'mode: %s\n' % self.DebugFormatInt32(self.mode_)
        return res