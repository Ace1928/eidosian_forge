from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class IndexPostfix(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = None
    has_before_ = 0
    before_ = 1
    has_before_ascending_ = 0
    before_ascending_ = 0

    def __init__(self, contents=None):
        self.index_value_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def index_value_size(self):
        return len(self.index_value_)

    def index_value_list(self):
        return self.index_value_

    def index_value(self, i):
        return self.index_value_[i]

    def mutable_index_value(self, i):
        return self.index_value_[i]

    def add_index_value(self):
        x = IndexPostfix_IndexValue()
        self.index_value_.append(x)
        return x

    def clear_index_value(self):
        self.index_value_ = []

    def key(self):
        if self.key_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.key_ is None:
                    self.key_ = Reference()
            finally:
                self.lazy_init_lock_.release()
        return self.key_

    def mutable_key(self):
        self.has_key_ = 1
        return self.key()

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            if self.key_ is not None:
                self.key_.Clear()

    def has_key(self):
        return self.has_key_

    def before(self):
        return self.before_

    def set_before(self, x):
        self.has_before_ = 1
        self.before_ = x

    def clear_before(self):
        if self.has_before_:
            self.has_before_ = 0
            self.before_ = 1

    def has_before(self):
        return self.has_before_

    def before_ascending(self):
        return self.before_ascending_

    def set_before_ascending(self, x):
        self.has_before_ascending_ = 1
        self.before_ascending_ = x

    def clear_before_ascending(self):
        if self.has_before_ascending_:
            self.has_before_ascending_ = 0
            self.before_ascending_ = 0

    def has_before_ascending(self):
        return self.has_before_ascending_

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.index_value_size()):
            self.add_index_value().CopyFrom(x.index_value(i))
        if x.has_key():
            self.mutable_key().MergeFrom(x.key())
        if x.has_before():
            self.set_before(x.before())
        if x.has_before_ascending():
            self.set_before_ascending(x.before_ascending())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.index_value_) != len(x.index_value_):
            return 0
        for e1, e2 in zip(self.index_value_, x.index_value_):
            if e1 != e2:
                return 0
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_before_ != x.has_before_:
            return 0
        if self.has_before_ and self.before_ != x.before_:
            return 0
        if self.has_before_ascending_ != x.has_before_ascending_:
            return 0
        if self.has_before_ascending_ and self.before_ascending_ != x.before_ascending_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.index_value_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_key_ and (not self.key_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.index_value_)
        for i in range(len(self.index_value_)):
            n += self.lengthString(self.index_value_[i].ByteSize())
        if self.has_key_:
            n += 1 + self.lengthString(self.key_.ByteSize())
        if self.has_before_:
            n += 2
        if self.has_before_ascending_:
            n += 2
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.index_value_)
        for i in range(len(self.index_value_)):
            n += self.lengthString(self.index_value_[i].ByteSizePartial())
        if self.has_key_:
            n += 1 + self.lengthString(self.key_.ByteSizePartial())
        if self.has_before_:
            n += 2
        if self.has_before_ascending_:
            n += 2
        return n

    def Clear(self):
        self.clear_index_value()
        self.clear_key()
        self.clear_before()
        self.clear_before_ascending()

    def OutputUnchecked(self, out):
        for i in range(len(self.index_value_)):
            out.putVarInt32(10)
            out.putVarInt32(self.index_value_[i].ByteSize())
            self.index_value_[i].OutputUnchecked(out)
        if self.has_key_:
            out.putVarInt32(18)
            out.putVarInt32(self.key_.ByteSize())
            self.key_.OutputUnchecked(out)
        if self.has_before_:
            out.putVarInt32(24)
            out.putBoolean(self.before_)
        if self.has_before_ascending_:
            out.putVarInt32(32)
            out.putBoolean(self.before_ascending_)

    def OutputPartial(self, out):
        for i in range(len(self.index_value_)):
            out.putVarInt32(10)
            out.putVarInt32(self.index_value_[i].ByteSizePartial())
            self.index_value_[i].OutputPartial(out)
        if self.has_key_:
            out.putVarInt32(18)
            out.putVarInt32(self.key_.ByteSizePartial())
            self.key_.OutputPartial(out)
        if self.has_before_:
            out.putVarInt32(24)
            out.putBoolean(self.before_)
        if self.has_before_ascending_:
            out.putVarInt32(32)
            out.putBoolean(self.before_ascending_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_index_value().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_key().TryMerge(tmp)
                continue
            if tt == 24:
                self.set_before(d.getBoolean())
                continue
            if tt == 32:
                self.set_before_ascending(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.index_value_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'index_value%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_key_:
            res += prefix + 'key <\n'
            res += self.key_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_before_:
            res += prefix + 'before: %s\n' % self.DebugFormatBool(self.before_)
        if self.has_before_ascending_:
            res += prefix + 'before_ascending: %s\n' % self.DebugFormatBool(self.before_ascending_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex_value = 1
    kkey = 2
    kbefore = 3
    kbefore_ascending = 4
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index_value', 2: 'key', 3: 'before', 4: 'before_ascending'}, 4)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC}, 4, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.IndexPostfix'