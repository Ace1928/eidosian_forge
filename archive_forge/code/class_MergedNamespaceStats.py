from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MergedNamespaceStats(ProtocolBuffer.ProtocolMessage):
    has_hits_ = 0
    hits_ = 0
    has_misses_ = 0
    misses_ = 0
    has_byte_hits_ = 0
    byte_hits_ = 0
    has_items_ = 0
    items_ = 0
    has_bytes_ = 0
    bytes_ = 0
    has_oldest_item_age_ = 0
    oldest_item_age_ = 0

    def __init__(self, contents=None):
        self.hotkeys_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def hits(self):
        return self.hits_

    def set_hits(self, x):
        self.has_hits_ = 1
        self.hits_ = x

    def clear_hits(self):
        if self.has_hits_:
            self.has_hits_ = 0
            self.hits_ = 0

    def has_hits(self):
        return self.has_hits_

    def misses(self):
        return self.misses_

    def set_misses(self, x):
        self.has_misses_ = 1
        self.misses_ = x

    def clear_misses(self):
        if self.has_misses_:
            self.has_misses_ = 0
            self.misses_ = 0

    def has_misses(self):
        return self.has_misses_

    def byte_hits(self):
        return self.byte_hits_

    def set_byte_hits(self, x):
        self.has_byte_hits_ = 1
        self.byte_hits_ = x

    def clear_byte_hits(self):
        if self.has_byte_hits_:
            self.has_byte_hits_ = 0
            self.byte_hits_ = 0

    def has_byte_hits(self):
        return self.has_byte_hits_

    def items(self):
        return self.items_

    def set_items(self, x):
        self.has_items_ = 1
        self.items_ = x

    def clear_items(self):
        if self.has_items_:
            self.has_items_ = 0
            self.items_ = 0

    def has_items(self):
        return self.has_items_

    def bytes(self):
        return self.bytes_

    def set_bytes(self, x):
        self.has_bytes_ = 1
        self.bytes_ = x

    def clear_bytes(self):
        if self.has_bytes_:
            self.has_bytes_ = 0
            self.bytes_ = 0

    def has_bytes(self):
        return self.has_bytes_

    def oldest_item_age(self):
        return self.oldest_item_age_

    def set_oldest_item_age(self, x):
        self.has_oldest_item_age_ = 1
        self.oldest_item_age_ = x

    def clear_oldest_item_age(self):
        if self.has_oldest_item_age_:
            self.has_oldest_item_age_ = 0
            self.oldest_item_age_ = 0

    def has_oldest_item_age(self):
        return self.has_oldest_item_age_

    def hotkeys_size(self):
        return len(self.hotkeys_)

    def hotkeys_list(self):
        return self.hotkeys_

    def hotkeys(self, i):
        return self.hotkeys_[i]

    def mutable_hotkeys(self, i):
        return self.hotkeys_[i]

    def add_hotkeys(self):
        x = MemcacheHotKey()
        self.hotkeys_.append(x)
        return x

    def clear_hotkeys(self):
        self.hotkeys_ = []

    def MergeFrom(self, x):
        assert x is not self
        if x.has_hits():
            self.set_hits(x.hits())
        if x.has_misses():
            self.set_misses(x.misses())
        if x.has_byte_hits():
            self.set_byte_hits(x.byte_hits())
        if x.has_items():
            self.set_items(list(x.items()))
        if x.has_bytes():
            self.set_bytes(x.bytes())
        if x.has_oldest_item_age():
            self.set_oldest_item_age(x.oldest_item_age())
        for i in range(x.hotkeys_size()):
            self.add_hotkeys().CopyFrom(x.hotkeys(i))

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_hits_ != x.has_hits_:
            return 0
        if self.has_hits_ and self.hits_ != x.hits_:
            return 0
        if self.has_misses_ != x.has_misses_:
            return 0
        if self.has_misses_ and self.misses_ != x.misses_:
            return 0
        if self.has_byte_hits_ != x.has_byte_hits_:
            return 0
        if self.has_byte_hits_ and self.byte_hits_ != x.byte_hits_:
            return 0
        if self.has_items_ != x.has_items_:
            return 0
        if self.has_items_ and self.items_ != x.items_:
            return 0
        if self.has_bytes_ != x.has_bytes_:
            return 0
        if self.has_bytes_ and self.bytes_ != x.bytes_:
            return 0
        if self.has_oldest_item_age_ != x.has_oldest_item_age_:
            return 0
        if self.has_oldest_item_age_ and self.oldest_item_age_ != x.oldest_item_age_:
            return 0
        if len(self.hotkeys_) != len(x.hotkeys_):
            return 0
        for e1, e2 in zip(self.hotkeys_, x.hotkeys_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_hits_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: hits not set.')
        if not self.has_misses_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: misses not set.')
        if not self.has_byte_hits_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: byte_hits not set.')
        if not self.has_items_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: items not set.')
        if not self.has_bytes_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: bytes not set.')
        if not self.has_oldest_item_age_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: oldest_item_age not set.')
        for p in self.hotkeys_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.hits_)
        n += self.lengthVarInt64(self.misses_)
        n += self.lengthVarInt64(self.byte_hits_)
        n += self.lengthVarInt64(self.items_)
        n += self.lengthVarInt64(self.bytes_)
        n += 1 * len(self.hotkeys_)
        for i in range(len(self.hotkeys_)):
            n += self.lengthString(self.hotkeys_[i].ByteSize())
        return n + 10

    def ByteSizePartial(self):
        n = 0
        if self.has_hits_:
            n += 1
            n += self.lengthVarInt64(self.hits_)
        if self.has_misses_:
            n += 1
            n += self.lengthVarInt64(self.misses_)
        if self.has_byte_hits_:
            n += 1
            n += self.lengthVarInt64(self.byte_hits_)
        if self.has_items_:
            n += 1
            n += self.lengthVarInt64(self.items_)
        if self.has_bytes_:
            n += 1
            n += self.lengthVarInt64(self.bytes_)
        if self.has_oldest_item_age_:
            n += 5
        n += 1 * len(self.hotkeys_)
        for i in range(len(self.hotkeys_)):
            n += self.lengthString(self.hotkeys_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_hits()
        self.clear_misses()
        self.clear_byte_hits()
        self.clear_items()
        self.clear_bytes()
        self.clear_oldest_item_age()
        self.clear_hotkeys()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putVarUint64(self.hits_)
        out.putVarInt32(16)
        out.putVarUint64(self.misses_)
        out.putVarInt32(24)
        out.putVarUint64(self.byte_hits_)
        out.putVarInt32(32)
        out.putVarUint64(self.items_)
        out.putVarInt32(40)
        out.putVarUint64(self.bytes_)
        out.putVarInt32(53)
        out.put32(self.oldest_item_age_)
        for i in range(len(self.hotkeys_)):
            out.putVarInt32(58)
            out.putVarInt32(self.hotkeys_[i].ByteSize())
            self.hotkeys_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_hits_:
            out.putVarInt32(8)
            out.putVarUint64(self.hits_)
        if self.has_misses_:
            out.putVarInt32(16)
            out.putVarUint64(self.misses_)
        if self.has_byte_hits_:
            out.putVarInt32(24)
            out.putVarUint64(self.byte_hits_)
        if self.has_items_:
            out.putVarInt32(32)
            out.putVarUint64(self.items_)
        if self.has_bytes_:
            out.putVarInt32(40)
            out.putVarUint64(self.bytes_)
        if self.has_oldest_item_age_:
            out.putVarInt32(53)
            out.put32(self.oldest_item_age_)
        for i in range(len(self.hotkeys_)):
            out.putVarInt32(58)
            out.putVarInt32(self.hotkeys_[i].ByteSizePartial())
            self.hotkeys_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_hits(d.getVarUint64())
                continue
            if tt == 16:
                self.set_misses(d.getVarUint64())
                continue
            if tt == 24:
                self.set_byte_hits(d.getVarUint64())
                continue
            if tt == 32:
                self.set_items(d.getVarUint64())
                continue
            if tt == 40:
                self.set_bytes(d.getVarUint64())
                continue
            if tt == 53:
                self.set_oldest_item_age(d.get32())
                continue
            if tt == 58:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_hotkeys().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_hits_:
            res += prefix + 'hits: %s\n' % self.DebugFormatInt64(self.hits_)
        if self.has_misses_:
            res += prefix + 'misses: %s\n' % self.DebugFormatInt64(self.misses_)
        if self.has_byte_hits_:
            res += prefix + 'byte_hits: %s\n' % self.DebugFormatInt64(self.byte_hits_)
        if self.has_items_:
            res += prefix + 'items: %s\n' % self.DebugFormatInt64(self.items_)
        if self.has_bytes_:
            res += prefix + 'bytes: %s\n' % self.DebugFormatInt64(self.bytes_)
        if self.has_oldest_item_age_:
            res += prefix + 'oldest_item_age: %s\n' % self.DebugFormatFixed32(self.oldest_item_age_)
        cnt = 0
        for e in self.hotkeys_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'hotkeys%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    khits = 1
    kmisses = 2
    kbyte_hits = 3
    kitems = 4
    kbytes = 5
    koldest_item_age = 6
    khotkeys = 7
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'hits', 2: 'misses', 3: 'byte_hits', 4: 'items', 5: 'bytes', 6: 'oldest_item_age', 7: 'hotkeys'}, 7)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.FLOAT, 7: ProtocolBuffer.Encoder.STRING}, 7, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MergedNamespaceStats'