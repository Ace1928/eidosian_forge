from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class PropertyValue_ReferenceValuePathElement(ProtocolBuffer.ProtocolMessage):
    has_type_ = 0
    type_ = ''
    has_id_ = 0
    id_ = 0
    has_name_ = 0
    name_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def type(self):
        return self.type_

    def set_type(self, x):
        self.has_type_ = 1
        self.type_ = x

    def clear_type(self):
        if self.has_type_:
            self.has_type_ = 0
            self.type_ = ''

    def has_type(self):
        return self.has_type_

    def id(self):
        return self.id_

    def set_id(self, x):
        self.has_id_ = 1
        self.id_ = x

    def clear_id(self):
        if self.has_id_:
            self.has_id_ = 0
            self.id_ = 0

    def has_id(self):
        return self.has_id_

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

    def MergeFrom(self, x):
        assert x is not self
        if x.has_type():
            self.set_type(x.type())
        if x.has_id():
            self.set_id(x.id())
        if x.has_name():
            self.set_name(x.name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_type_ != x.has_type_:
            return 0
        if self.has_type_ and self.type_ != x.type_:
            return 0
        if self.has_id_ != x.has_id_:
            return 0
        if self.has_id_ and self.id_ != x.id_:
            return 0
        if self.has_name_ != x.has_name_:
            return 0
        if self.has_name_ and self.name_ != x.name_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_type_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: type not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.type_))
        if self.has_id_:
            n += 2 + self.lengthVarInt64(self.id_)
        if self.has_name_:
            n += 2 + self.lengthString(len(self.name_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_type_:
            n += 1
            n += self.lengthString(len(self.type_))
        if self.has_id_:
            n += 2 + self.lengthVarInt64(self.id_)
        if self.has_name_:
            n += 2 + self.lengthString(len(self.name_))
        return n

    def Clear(self):
        self.clear_type()
        self.clear_id()
        self.clear_name()

    def OutputUnchecked(self, out):
        out.putVarInt32(122)
        out.putPrefixedString(self.type_)
        if self.has_id_:
            out.putVarInt32(128)
            out.putVarInt64(self.id_)
        if self.has_name_:
            out.putVarInt32(138)
            out.putPrefixedString(self.name_)

    def OutputPartial(self, out):
        if self.has_type_:
            out.putVarInt32(122)
            out.putPrefixedString(self.type_)
        if self.has_id_:
            out.putVarInt32(128)
            out.putVarInt64(self.id_)
        if self.has_name_:
            out.putVarInt32(138)
            out.putPrefixedString(self.name_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 116:
                break
            if tt == 122:
                self.set_type(d.getPrefixedString())
                continue
            if tt == 128:
                self.set_id(d.getVarInt64())
                continue
            if tt == 138:
                self.set_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_type_:
            res += prefix + 'type: %s\n' % self.DebugFormatString(self.type_)
        if self.has_id_:
            res += prefix + 'id: %s\n' % self.DebugFormatInt64(self.id_)
        if self.has_name_:
            res += prefix + 'name: %s\n' % self.DebugFormatString(self.name_)
        return res