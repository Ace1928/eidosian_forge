from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class Key_PathElement(ProtocolBuffer.ProtocolMessage):
    has_kind_ = 0
    kind_ = ''
    has_id_ = 0
    id_ = 0
    has_name_ = 0
    name_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def kind(self):
        return self.kind_

    def set_kind(self, x):
        self.has_kind_ = 1
        self.kind_ = x

    def clear_kind(self):
        if self.has_kind_:
            self.has_kind_ = 0
            self.kind_ = ''

    def has_kind(self):
        return self.has_kind_

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
        if x.has_kind():
            self.set_kind(x.kind())
        if x.has_id():
            self.set_id(x.id())
        if x.has_name():
            self.set_name(x.name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_kind_ != x.has_kind_:
            return 0
        if self.has_kind_ and self.kind_ != x.kind_:
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
        if not self.has_kind_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: kind not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.kind_))
        if self.has_id_:
            n += 1 + self.lengthVarInt64(self.id_)
        if self.has_name_:
            n += 1 + self.lengthString(len(self.name_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_kind_:
            n += 1
            n += self.lengthString(len(self.kind_))
        if self.has_id_:
            n += 1 + self.lengthVarInt64(self.id_)
        if self.has_name_:
            n += 1 + self.lengthString(len(self.name_))
        return n

    def Clear(self):
        self.clear_kind()
        self.clear_id()
        self.clear_name()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.kind_)
        if self.has_id_:
            out.putVarInt32(16)
            out.putVarInt64(self.id_)
        if self.has_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.name_)

    def OutputPartial(self, out):
        if self.has_kind_:
            out.putVarInt32(10)
            out.putPrefixedString(self.kind_)
        if self.has_id_:
            out.putVarInt32(16)
            out.putVarInt64(self.id_)
        if self.has_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.name_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_kind(d.getPrefixedString())
                continue
            if tt == 16:
                self.set_id(d.getVarInt64())
                continue
            if tt == 26:
                self.set_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_kind_:
            res += prefix + 'kind: %s\n' % self.DebugFormatString(self.kind_)
        if self.has_id_:
            res += prefix + 'id: %s\n' % self.DebugFormatInt64(self.id_)
        if self.has_name_:
            res += prefix + 'name: %s\n' % self.DebugFormatString(self.name_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkind = 1
    kid = 2
    kname = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'kind', 2: 'id', 3: 'name'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.STRING}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.Key_PathElement'