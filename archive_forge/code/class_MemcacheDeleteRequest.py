from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheDeleteRequest(ProtocolBuffer.ProtocolMessage):
    has_name_space_ = 0
    name_space_ = ''
    has_override_ = 0
    override_ = None

    def __init__(self, contents=None):
        self.item_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def item_size(self):
        return len(self.item_)

    def item_list(self):
        return self.item_

    def item(self, i):
        return self.item_[i]

    def mutable_item(self, i):
        return self.item_[i]

    def add_item(self):
        x = MemcacheDeleteRequest_Item()
        self.item_.append(x)
        return x

    def clear_item(self):
        self.item_ = []

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
        for i in range(x.item_size()):
            self.add_item().CopyFrom(x.item(i))
        if x.has_name_space():
            self.set_name_space(x.name_space())
        if x.has_override():
            self.mutable_override().MergeFrom(x.override())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.item_) != len(x.item_):
            return 0
        for e1, e2 in zip(self.item_, x.item_):
            if e1 != e2:
                return 0
        if self.has_name_space_ != x.has_name_space_:
            return 0
        if self.has_name_space_ and self.name_space_ != x.name_space_:
            return 0
        if self.has_override_ != x.has_override_:
            return 0
        if self.has_override_ and self.override_ != x.override_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.item_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_override_ and (not self.override_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.item_[i].ByteSize()
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.item_[i].ByteSizePartial()
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        if self.has_override_:
            n += 1 + self.lengthString(self.override_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_item()
        self.clear_name_space()
        self.clear_override()

    def OutputUnchecked(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(11)
            self.item_[i].OutputUnchecked(out)
            out.putVarInt32(12)
        if self.has_name_space_:
            out.putVarInt32(34)
            out.putPrefixedString(self.name_space_)
        if self.has_override_:
            out.putVarInt32(42)
            out.putVarInt32(self.override_.ByteSize())
            self.override_.OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(11)
            self.item_[i].OutputPartial(out)
            out.putVarInt32(12)
        if self.has_name_space_:
            out.putVarInt32(34)
            out.putPrefixedString(self.name_space_)
        if self.has_override_:
            out.putVarInt32(42)
            out.putVarInt32(self.override_.ByteSizePartial())
            self.override_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_item().TryMerge(d)
                continue
            if tt == 34:
                self.set_name_space(d.getPrefixedString())
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
        for e in self.item_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Item%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        if self.has_name_space_:
            res += prefix + 'name_space: %s\n' % self.DebugFormatString(self.name_space_)
        if self.has_override_:
            res += prefix + 'override <\n'
            res += self.override_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kItemGroup = 1
    kItemkey = 2
    kItemdelete_time = 3
    kname_space = 4
    koverride = 5
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'Item', 2: 'key', 3: 'delete_time', 4: 'name_space', 5: 'override'}, 5)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.FLOAT, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.STRING}, 5, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheDeleteRequest'