from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class CompositeProperty(ProtocolBuffer.ProtocolMessage):
    has_index_id_ = 0
    index_id_ = 0

    def __init__(self, contents=None):
        self.value_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def index_id(self):
        return self.index_id_

    def set_index_id(self, x):
        self.has_index_id_ = 1
        self.index_id_ = x

    def clear_index_id(self):
        if self.has_index_id_:
            self.has_index_id_ = 0
            self.index_id_ = 0

    def has_index_id(self):
        return self.has_index_id_

    def value_size(self):
        return len(self.value_)

    def value_list(self):
        return self.value_

    def value(self, i):
        return self.value_[i]

    def set_value(self, i, x):
        self.value_[i] = x

    def add_value(self, x):
        self.value_.append(x)

    def clear_value(self):
        self.value_ = []

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_id():
            self.set_index_id(x.index_id())
        for i in range(x.value_size()):
            self.add_value(x.value(i))

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_id_ != x.has_index_id_:
            return 0
        if self.has_index_id_ and self.index_id_ != x.index_id_:
            return 0
        if len(self.value_) != len(x.value_):
            return 0
        for e1, e2 in zip(self.value_, x.value_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_index_id_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: index_id not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.index_id_)
        n += 1 * len(self.value_)
        for i in range(len(self.value_)):
            n += self.lengthString(len(self.value_[i]))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_index_id_:
            n += 1
            n += self.lengthVarInt64(self.index_id_)
        n += 1 * len(self.value_)
        for i in range(len(self.value_)):
            n += self.lengthString(len(self.value_[i]))
        return n

    def Clear(self):
        self.clear_index_id()
        self.clear_value()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putVarInt64(self.index_id_)
        for i in range(len(self.value_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.value_[i])

    def OutputPartial(self, out):
        if self.has_index_id_:
            out.putVarInt32(8)
            out.putVarInt64(self.index_id_)
        for i in range(len(self.value_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.value_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_index_id(d.getVarInt64())
                continue
            if tt == 18:
                self.add_value(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_id_:
            res += prefix + 'index_id: %s\n' % self.DebugFormatInt64(self.index_id_)
        cnt = 0
        for e in self.value_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'value%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex_id = 1
    kvalue = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index_id', 2: 'value'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.CompositeProperty'