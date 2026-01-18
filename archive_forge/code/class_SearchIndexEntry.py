from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class SearchIndexEntry(ProtocolBuffer.ProtocolMessage):
    has_index_id_ = 0
    index_id_ = 0
    has_write_division_family_ = 0
    write_division_family_ = ''
    has_fingerprint_1999_ = 0
    fingerprint_1999_ = 0
    has_fingerprint_2011_ = 0
    fingerprint_2011_ = 0

    def __init__(self, contents=None):
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

    def write_division_family(self):
        return self.write_division_family_

    def set_write_division_family(self, x):
        self.has_write_division_family_ = 1
        self.write_division_family_ = x

    def clear_write_division_family(self):
        if self.has_write_division_family_:
            self.has_write_division_family_ = 0
            self.write_division_family_ = ''

    def has_write_division_family(self):
        return self.has_write_division_family_

    def fingerprint_1999(self):
        return self.fingerprint_1999_

    def set_fingerprint_1999(self, x):
        self.has_fingerprint_1999_ = 1
        self.fingerprint_1999_ = x

    def clear_fingerprint_1999(self):
        if self.has_fingerprint_1999_:
            self.has_fingerprint_1999_ = 0
            self.fingerprint_1999_ = 0

    def has_fingerprint_1999(self):
        return self.has_fingerprint_1999_

    def fingerprint_2011(self):
        return self.fingerprint_2011_

    def set_fingerprint_2011(self, x):
        self.has_fingerprint_2011_ = 1
        self.fingerprint_2011_ = x

    def clear_fingerprint_2011(self):
        if self.has_fingerprint_2011_:
            self.has_fingerprint_2011_ = 0
            self.fingerprint_2011_ = 0

    def has_fingerprint_2011(self):
        return self.has_fingerprint_2011_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_id():
            self.set_index_id(x.index_id())
        if x.has_write_division_family():
            self.set_write_division_family(x.write_division_family())
        if x.has_fingerprint_1999():
            self.set_fingerprint_1999(x.fingerprint_1999())
        if x.has_fingerprint_2011():
            self.set_fingerprint_2011(x.fingerprint_2011())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_id_ != x.has_index_id_:
            return 0
        if self.has_index_id_ and self.index_id_ != x.index_id_:
            return 0
        if self.has_write_division_family_ != x.has_write_division_family_:
            return 0
        if self.has_write_division_family_ and self.write_division_family_ != x.write_division_family_:
            return 0
        if self.has_fingerprint_1999_ != x.has_fingerprint_1999_:
            return 0
        if self.has_fingerprint_1999_ and self.fingerprint_1999_ != x.fingerprint_1999_:
            return 0
        if self.has_fingerprint_2011_ != x.has_fingerprint_2011_:
            return 0
        if self.has_fingerprint_2011_ and self.fingerprint_2011_ != x.fingerprint_2011_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_index_id_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: index_id not set.')
        if not self.has_write_division_family_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: write_division_family not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.index_id_)
        n += self.lengthString(len(self.write_division_family_))
        if self.has_fingerprint_1999_:
            n += 9
        if self.has_fingerprint_2011_:
            n += 9
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_index_id_:
            n += 1
            n += self.lengthVarInt64(self.index_id_)
        if self.has_write_division_family_:
            n += 1
            n += self.lengthString(len(self.write_division_family_))
        if self.has_fingerprint_1999_:
            n += 9
        if self.has_fingerprint_2011_:
            n += 9
        return n

    def Clear(self):
        self.clear_index_id()
        self.clear_write_division_family()
        self.clear_fingerprint_1999()
        self.clear_fingerprint_2011()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putVarInt64(self.index_id_)
        out.putVarInt32(18)
        out.putPrefixedString(self.write_division_family_)
        if self.has_fingerprint_1999_:
            out.putVarInt32(25)
            out.put64(self.fingerprint_1999_)
        if self.has_fingerprint_2011_:
            out.putVarInt32(33)
            out.put64(self.fingerprint_2011_)

    def OutputPartial(self, out):
        if self.has_index_id_:
            out.putVarInt32(8)
            out.putVarInt64(self.index_id_)
        if self.has_write_division_family_:
            out.putVarInt32(18)
            out.putPrefixedString(self.write_division_family_)
        if self.has_fingerprint_1999_:
            out.putVarInt32(25)
            out.put64(self.fingerprint_1999_)
        if self.has_fingerprint_2011_:
            out.putVarInt32(33)
            out.put64(self.fingerprint_2011_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_index_id(d.getVarInt64())
                continue
            if tt == 18:
                self.set_write_division_family(d.getPrefixedString())
                continue
            if tt == 25:
                self.set_fingerprint_1999(d.get64())
                continue
            if tt == 33:
                self.set_fingerprint_2011(d.get64())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_id_:
            res += prefix + 'index_id: %s\n' % self.DebugFormatInt64(self.index_id_)
        if self.has_write_division_family_:
            res += prefix + 'write_division_family: %s\n' % self.DebugFormatString(self.write_division_family_)
        if self.has_fingerprint_1999_:
            res += prefix + 'fingerprint_1999: %s\n' % self.DebugFormatFixed64(self.fingerprint_1999_)
        if self.has_fingerprint_2011_:
            res += prefix + 'fingerprint_2011: %s\n' % self.DebugFormatFixed64(self.fingerprint_2011_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex_id = 1
    kwrite_division_family = 2
    kfingerprint_1999 = 3
    kfingerprint_2011 = 4
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index_id', 2: 'write_division_family', 3: 'fingerprint_1999', 4: 'fingerprint_2011'}, 4)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.DOUBLE, 4: ProtocolBuffer.Encoder.DOUBLE}, 4, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.SearchIndexEntry'