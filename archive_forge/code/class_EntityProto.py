from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class EntityProto(ProtocolBuffer.ProtocolMessage):
    GD_CONTACT = 1
    GD_EVENT = 2
    GD_MESSAGE = 3
    _Kind_NAMES = {1: 'GD_CONTACT', 2: 'GD_EVENT', 3: 'GD_MESSAGE'}

    def Kind_Name(cls, x):
        return cls._Kind_NAMES.get(x, '')
    Kind_Name = classmethod(Kind_Name)
    has_key_ = 0
    has_entity_group_ = 0
    has_owner_ = 0
    owner_ = None
    has_kind_ = 0
    kind_ = 0
    has_kind_uri_ = 0
    kind_uri_ = ''

    def __init__(self, contents=None):
        self.key_ = Reference()
        self.entity_group_ = Path()
        self.property_ = []
        self.raw_property_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def key(self):
        return self.key_

    def mutable_key(self):
        self.has_key_ = 1
        return self.key_

    def clear_key(self):
        self.has_key_ = 0
        self.key_.Clear()

    def has_key(self):
        return self.has_key_

    def entity_group(self):
        return self.entity_group_

    def mutable_entity_group(self):
        self.has_entity_group_ = 1
        return self.entity_group_

    def clear_entity_group(self):
        self.has_entity_group_ = 0
        self.entity_group_.Clear()

    def has_entity_group(self):
        return self.has_entity_group_

    def owner(self):
        if self.owner_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.owner_ is None:
                    self.owner_ = User()
            finally:
                self.lazy_init_lock_.release()
        return self.owner_

    def mutable_owner(self):
        self.has_owner_ = 1
        return self.owner()

    def clear_owner(self):
        if self.has_owner_:
            self.has_owner_ = 0
            if self.owner_ is not None:
                self.owner_.Clear()

    def has_owner(self):
        return self.has_owner_

    def kind(self):
        return self.kind_

    def set_kind(self, x):
        self.has_kind_ = 1
        self.kind_ = x

    def clear_kind(self):
        if self.has_kind_:
            self.has_kind_ = 0
            self.kind_ = 0

    def has_kind(self):
        return self.has_kind_

    def kind_uri(self):
        return self.kind_uri_

    def set_kind_uri(self, x):
        self.has_kind_uri_ = 1
        self.kind_uri_ = x

    def clear_kind_uri(self):
        if self.has_kind_uri_:
            self.has_kind_uri_ = 0
            self.kind_uri_ = ''

    def has_kind_uri(self):
        return self.has_kind_uri_

    def property_size(self):
        return len(self.property_)

    def property_list(self):
        return self.property_

    def property(self, i):
        return self.property_[i]

    def mutable_property(self, i):
        return self.property_[i]

    def add_property(self):
        x = Property()
        self.property_.append(x)
        return x

    def clear_property(self):
        self.property_ = []

    def raw_property_size(self):
        return len(self.raw_property_)

    def raw_property_list(self):
        return self.raw_property_

    def raw_property(self, i):
        return self.raw_property_[i]

    def mutable_raw_property(self, i):
        return self.raw_property_[i]

    def add_raw_property(self):
        x = Property()
        self.raw_property_.append(x)
        return x

    def clear_raw_property(self):
        self.raw_property_ = []

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.mutable_key().MergeFrom(x.key())
        if x.has_entity_group():
            self.mutable_entity_group().MergeFrom(x.entity_group())
        if x.has_owner():
            self.mutable_owner().MergeFrom(x.owner())
        if x.has_kind():
            self.set_kind(x.kind())
        if x.has_kind_uri():
            self.set_kind_uri(x.kind_uri())
        for i in range(x.property_size()):
            self.add_property().CopyFrom(x.property(i))
        for i in range(x.raw_property_size()):
            self.add_raw_property().CopyFrom(x.raw_property(i))

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_entity_group_ != x.has_entity_group_:
            return 0
        if self.has_entity_group_ and self.entity_group_ != x.entity_group_:
            return 0
        if self.has_owner_ != x.has_owner_:
            return 0
        if self.has_owner_ and self.owner_ != x.owner_:
            return 0
        if self.has_kind_ != x.has_kind_:
            return 0
        if self.has_kind_ and self.kind_ != x.kind_:
            return 0
        if self.has_kind_uri_ != x.has_kind_uri_:
            return 0
        if self.has_kind_uri_ and self.kind_uri_ != x.kind_uri_:
            return 0
        if len(self.property_) != len(x.property_):
            return 0
        for e1, e2 in zip(self.property_, x.property_):
            if e1 != e2:
                return 0
        if len(self.raw_property_) != len(x.raw_property_):
            return 0
        for e1, e2 in zip(self.raw_property_, x.raw_property_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        elif not self.key_.IsInitialized(debug_strs):
            initialized = 0
        if not self.has_entity_group_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: entity_group not set.')
        elif not self.entity_group_.IsInitialized(debug_strs):
            initialized = 0
        if self.has_owner_ and (not self.owner_.IsInitialized(debug_strs)):
            initialized = 0
        for p in self.property_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.raw_property_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.key_.ByteSize())
        n += self.lengthString(self.entity_group_.ByteSize())
        if self.has_owner_:
            n += 2 + self.lengthString(self.owner_.ByteSize())
        if self.has_kind_:
            n += 1 + self.lengthVarInt64(self.kind_)
        if self.has_kind_uri_:
            n += 1 + self.lengthString(len(self.kind_uri_))
        n += 1 * len(self.property_)
        for i in range(len(self.property_)):
            n += self.lengthString(self.property_[i].ByteSize())
        n += 1 * len(self.raw_property_)
        for i in range(len(self.raw_property_)):
            n += self.lengthString(self.raw_property_[i].ByteSize())
        return n + 3

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(self.key_.ByteSizePartial())
        if self.has_entity_group_:
            n += 2
            n += self.lengthString(self.entity_group_.ByteSizePartial())
        if self.has_owner_:
            n += 2 + self.lengthString(self.owner_.ByteSizePartial())
        if self.has_kind_:
            n += 1 + self.lengthVarInt64(self.kind_)
        if self.has_kind_uri_:
            n += 1 + self.lengthString(len(self.kind_uri_))
        n += 1 * len(self.property_)
        for i in range(len(self.property_)):
            n += self.lengthString(self.property_[i].ByteSizePartial())
        n += 1 * len(self.raw_property_)
        for i in range(len(self.raw_property_)):
            n += self.lengthString(self.raw_property_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_key()
        self.clear_entity_group()
        self.clear_owner()
        self.clear_kind()
        self.clear_kind_uri()
        self.clear_property()
        self.clear_raw_property()

    def OutputUnchecked(self, out):
        if self.has_kind_:
            out.putVarInt32(32)
            out.putVarInt32(self.kind_)
        if self.has_kind_uri_:
            out.putVarInt32(42)
            out.putPrefixedString(self.kind_uri_)
        out.putVarInt32(106)
        out.putVarInt32(self.key_.ByteSize())
        self.key_.OutputUnchecked(out)
        for i in range(len(self.property_)):
            out.putVarInt32(114)
            out.putVarInt32(self.property_[i].ByteSize())
            self.property_[i].OutputUnchecked(out)
        for i in range(len(self.raw_property_)):
            out.putVarInt32(122)
            out.putVarInt32(self.raw_property_[i].ByteSize())
            self.raw_property_[i].OutputUnchecked(out)
        out.putVarInt32(130)
        out.putVarInt32(self.entity_group_.ByteSize())
        self.entity_group_.OutputUnchecked(out)
        if self.has_owner_:
            out.putVarInt32(138)
            out.putVarInt32(self.owner_.ByteSize())
            self.owner_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_kind_:
            out.putVarInt32(32)
            out.putVarInt32(self.kind_)
        if self.has_kind_uri_:
            out.putVarInt32(42)
            out.putPrefixedString(self.kind_uri_)
        if self.has_key_:
            out.putVarInt32(106)
            out.putVarInt32(self.key_.ByteSizePartial())
            self.key_.OutputPartial(out)
        for i in range(len(self.property_)):
            out.putVarInt32(114)
            out.putVarInt32(self.property_[i].ByteSizePartial())
            self.property_[i].OutputPartial(out)
        for i in range(len(self.raw_property_)):
            out.putVarInt32(122)
            out.putVarInt32(self.raw_property_[i].ByteSizePartial())
            self.raw_property_[i].OutputPartial(out)
        if self.has_entity_group_:
            out.putVarInt32(130)
            out.putVarInt32(self.entity_group_.ByteSizePartial())
            self.entity_group_.OutputPartial(out)
        if self.has_owner_:
            out.putVarInt32(138)
            out.putVarInt32(self.owner_.ByteSizePartial())
            self.owner_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 32:
                self.set_kind(d.getVarInt32())
                continue
            if tt == 42:
                self.set_kind_uri(d.getPrefixedString())
                continue
            if tt == 106:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_key().TryMerge(tmp)
                continue
            if tt == 114:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_property().TryMerge(tmp)
                continue
            if tt == 122:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_raw_property().TryMerge(tmp)
                continue
            if tt == 130:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_entity_group().TryMerge(tmp)
                continue
            if tt == 138:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_owner().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key <\n'
            res += self.key_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_entity_group_:
            res += prefix + 'entity_group <\n'
            res += self.entity_group_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_owner_:
            res += prefix + 'owner <\n'
            res += self.owner_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_kind_:
            res += prefix + 'kind: %s\n' % self.DebugFormatInt32(self.kind_)
        if self.has_kind_uri_:
            res += prefix + 'kind_uri: %s\n' % self.DebugFormatString(self.kind_uri_)
        cnt = 0
        for e in self.property_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'property%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.raw_property_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'raw_property%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 13
    kentity_group = 16
    kowner = 17
    kkind = 4
    kkind_uri = 5
    kproperty = 14
    kraw_property = 15
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 4: 'kind', 5: 'kind_uri', 13: 'key', 14: 'property', 15: 'raw_property', 16: 'entity_group', 17: 'owner'}, 17)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STRING, 13: ProtocolBuffer.Encoder.STRING, 14: ProtocolBuffer.Encoder.STRING, 15: ProtocolBuffer.Encoder.STRING, 16: ProtocolBuffer.Encoder.STRING, 17: ProtocolBuffer.Encoder.STRING}, 17, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.EntityProto'