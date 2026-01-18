from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class IndexPostfix_IndexValue(ProtocolBuffer.ProtocolMessage):
    has_property_name_ = 0
    property_name_ = ''
    has_value_ = 0

    def __init__(self, contents=None):
        self.value_ = PropertyValue()
        if contents is not None:
            self.MergeFromString(contents)

    def property_name(self):
        return self.property_name_

    def set_property_name(self, x):
        self.has_property_name_ = 1
        self.property_name_ = x

    def clear_property_name(self):
        if self.has_property_name_:
            self.has_property_name_ = 0
            self.property_name_ = ''

    def has_property_name(self):
        return self.has_property_name_

    def value(self):
        return self.value_

    def mutable_value(self):
        self.has_value_ = 1
        return self.value_

    def clear_value(self):
        self.has_value_ = 0
        self.value_.Clear()

    def has_value(self):
        return self.has_value_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_property_name():
            self.set_property_name(x.property_name())
        if x.has_value():
            self.mutable_value().MergeFrom(x.value())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_property_name_ != x.has_property_name_:
            return 0
        if self.has_property_name_ and self.property_name_ != x.property_name_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_property_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: property_name not set.')
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        elif not self.value_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.property_name_))
        n += self.lengthString(self.value_.ByteSize())
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_property_name_:
            n += 1
            n += self.lengthString(len(self.property_name_))
        if self.has_value_:
            n += 1
            n += self.lengthString(self.value_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_property_name()
        self.clear_value()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.property_name_)
        out.putVarInt32(18)
        out.putVarInt32(self.value_.ByteSize())
        self.value_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_property_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.property_name_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putVarInt32(self.value_.ByteSizePartial())
            self.value_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_property_name(d.getPrefixedString())
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_value().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_property_name_:
            res += prefix + 'property_name: %s\n' % self.DebugFormatString(self.property_name_)
        if self.has_value_:
            res += prefix + 'value <\n'
            res += self.value_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kproperty_name = 1
    kvalue = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'property_name', 2: 'value'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.IndexPostfix_IndexValue'