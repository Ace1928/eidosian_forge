from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheHotKey(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_qps_ = 0
    qps_ = 0.0
    has_name_space_ = 0
    name_space_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def key(self):
        return self.key_

    def set_key(self, x):
        self.has_key_ = 1
        self.key_ = x

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            self.key_ = ''

    def has_key(self):
        return self.has_key_

    def qps(self):
        return self.qps_

    def set_qps(self, x):
        self.has_qps_ = 1
        self.qps_ = x

    def clear_qps(self):
        if self.has_qps_:
            self.has_qps_ = 0
            self.qps_ = 0.0

    def has_qps(self):
        return self.has_qps_

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

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_qps():
            self.set_qps(x.qps())
        if x.has_name_space():
            self.set_name_space(x.name_space())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_qps_ != x.has_qps_:
            return 0
        if self.has_qps_ and self.qps_ != x.qps_:
            return 0
        if self.has_name_space_ != x.has_name_space_:
            return 0
        if self.has_name_space_ and self.name_space_ != x.name_space_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        if not self.has_qps_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: qps not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        return n + 10

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_qps_:
            n += 9
        if self.has_name_space_:
            n += 1 + self.lengthString(len(self.name_space_))
        return n

    def Clear(self):
        self.clear_key()
        self.clear_qps()
        self.clear_name_space()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.key_)
        out.putVarInt32(17)
        out.putDouble(self.qps_)
        if self.has_name_space_:
            out.putVarInt32(26)
            out.putPrefixedString(self.name_space_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(10)
            out.putPrefixedString(self.key_)
        if self.has_qps_:
            out.putVarInt32(17)
            out.putDouble(self.qps_)
        if self.has_name_space_:
            out.putVarInt32(26)
            out.putPrefixedString(self.name_space_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 17:
                self.set_qps(d.getDouble())
                continue
            if tt == 26:
                self.set_name_space(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_qps_:
            res += prefix + 'qps: %s\n' % self.DebugFormat(self.qps_)
        if self.has_name_space_:
            res += prefix + 'name_space: %s\n' % self.DebugFormatString(self.name_space_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kqps = 2
    kname_space = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'qps', 3: 'name_space'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.DOUBLE, 3: ProtocolBuffer.Encoder.STRING}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheHotKey'