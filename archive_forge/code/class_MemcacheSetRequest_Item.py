from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheSetRequest_Item(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_value_ = 0
    value_ = ''
    has_flags_ = 0
    flags_ = 0
    has_set_policy_ = 0
    set_policy_ = 1
    has_expiration_time_ = 0
    expiration_time_ = 0
    has_cas_id_ = 0
    cas_id_ = 0
    has_for_cas_ = 0
    for_cas_ = 0

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

    def value(self):
        return self.value_

    def set_value(self, x):
        self.has_value_ = 1
        self.value_ = x

    def clear_value(self):
        if self.has_value_:
            self.has_value_ = 0
            self.value_ = ''

    def has_value(self):
        return self.has_value_

    def flags(self):
        return self.flags_

    def set_flags(self, x):
        self.has_flags_ = 1
        self.flags_ = x

    def clear_flags(self):
        if self.has_flags_:
            self.has_flags_ = 0
            self.flags_ = 0

    def has_flags(self):
        return self.has_flags_

    def set_policy(self):
        return self.set_policy_

    def set_set_policy(self, x):
        self.has_set_policy_ = 1
        self.set_policy_ = x

    def clear_set_policy(self):
        if self.has_set_policy_:
            self.has_set_policy_ = 0
            self.set_policy_ = 1

    def has_set_policy(self):
        return self.has_set_policy_

    def expiration_time(self):
        return self.expiration_time_

    def set_expiration_time(self, x):
        self.has_expiration_time_ = 1
        self.expiration_time_ = x

    def clear_expiration_time(self):
        if self.has_expiration_time_:
            self.has_expiration_time_ = 0
            self.expiration_time_ = 0

    def has_expiration_time(self):
        return self.has_expiration_time_

    def cas_id(self):
        return self.cas_id_

    def set_cas_id(self, x):
        self.has_cas_id_ = 1
        self.cas_id_ = x

    def clear_cas_id(self):
        if self.has_cas_id_:
            self.has_cas_id_ = 0
            self.cas_id_ = 0

    def has_cas_id(self):
        return self.has_cas_id_

    def for_cas(self):
        return self.for_cas_

    def set_for_cas(self, x):
        self.has_for_cas_ = 1
        self.for_cas_ = x

    def clear_for_cas(self):
        if self.has_for_cas_:
            self.has_for_cas_ = 0
            self.for_cas_ = 0

    def has_for_cas(self):
        return self.has_for_cas_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_value():
            self.set_value(x.value())
        if x.has_flags():
            self.set_flags(x.flags())
        if x.has_set_policy():
            self.set_set_policy(x.set_policy())
        if x.has_expiration_time():
            self.set_expiration_time(x.expiration_time())
        if x.has_cas_id():
            self.set_cas_id(x.cas_id())
        if x.has_for_cas():
            self.set_for_cas(x.for_cas())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        if self.has_flags_ != x.has_flags_:
            return 0
        if self.has_flags_ and self.flags_ != x.flags_:
            return 0
        if self.has_set_policy_ != x.has_set_policy_:
            return 0
        if self.has_set_policy_ and self.set_policy_ != x.set_policy_:
            return 0
        if self.has_expiration_time_ != x.has_expiration_time_:
            return 0
        if self.has_expiration_time_ and self.expiration_time_ != x.expiration_time_:
            return 0
        if self.has_cas_id_ != x.has_cas_id_:
            return 0
        if self.has_cas_id_ and self.cas_id_ != x.cas_id_:
            return 0
        if self.has_for_cas_ != x.has_for_cas_:
            return 0
        if self.has_for_cas_ and self.for_cas_ != x.for_cas_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        n += self.lengthString(len(self.value_))
        if self.has_flags_:
            n += 5
        if self.has_set_policy_:
            n += 1 + self.lengthVarInt64(self.set_policy_)
        if self.has_expiration_time_:
            n += 5
        if self.has_cas_id_:
            n += 9
        if self.has_for_cas_:
            n += 2
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_value_:
            n += 1
            n += self.lengthString(len(self.value_))
        if self.has_flags_:
            n += 5
        if self.has_set_policy_:
            n += 1 + self.lengthVarInt64(self.set_policy_)
        if self.has_expiration_time_:
            n += 5
        if self.has_cas_id_:
            n += 9
        if self.has_for_cas_:
            n += 2
        return n

    def Clear(self):
        self.clear_key()
        self.clear_value()
        self.clear_flags()
        self.clear_set_policy()
        self.clear_expiration_time()
        self.clear_cas_id()
        self.clear_for_cas()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.key_)
        out.putVarInt32(26)
        out.putPrefixedString(self.value_)
        if self.has_flags_:
            out.putVarInt32(37)
            out.put32(self.flags_)
        if self.has_set_policy_:
            out.putVarInt32(40)
            out.putVarInt32(self.set_policy_)
        if self.has_expiration_time_:
            out.putVarInt32(53)
            out.put32(self.expiration_time_)
        if self.has_cas_id_:
            out.putVarInt32(65)
            out.put64(self.cas_id_)
        if self.has_for_cas_:
            out.putVarInt32(72)
            out.putBoolean(self.for_cas_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(18)
            out.putPrefixedString(self.key_)
        if self.has_value_:
            out.putVarInt32(26)
            out.putPrefixedString(self.value_)
        if self.has_flags_:
            out.putVarInt32(37)
            out.put32(self.flags_)
        if self.has_set_policy_:
            out.putVarInt32(40)
            out.putVarInt32(self.set_policy_)
        if self.has_expiration_time_:
            out.putVarInt32(53)
            out.put32(self.expiration_time_)
        if self.has_cas_id_:
            out.putVarInt32(65)
            out.put64(self.cas_id_)
        if self.has_for_cas_:
            out.putVarInt32(72)
            out.putBoolean(self.for_cas_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 26:
                self.set_value(d.getPrefixedString())
                continue
            if tt == 37:
                self.set_flags(d.get32())
                continue
            if tt == 40:
                self.set_set_policy(d.getVarInt32())
                continue
            if tt == 53:
                self.set_expiration_time(d.get32())
                continue
            if tt == 65:
                self.set_cas_id(d.get64())
                continue
            if tt == 72:
                self.set_for_cas(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_value_:
            res += prefix + 'value: %s\n' % self.DebugFormatString(self.value_)
        if self.has_flags_:
            res += prefix + 'flags: %s\n' % self.DebugFormatFixed32(self.flags_)
        if self.has_set_policy_:
            res += prefix + 'set_policy: %s\n' % self.DebugFormatInt32(self.set_policy_)
        if self.has_expiration_time_:
            res += prefix + 'expiration_time: %s\n' % self.DebugFormatFixed32(self.expiration_time_)
        if self.has_cas_id_:
            res += prefix + 'cas_id: %s\n' % self.DebugFormatFixed64(self.cas_id_)
        if self.has_for_cas_:
            res += prefix + 'for_cas: %s\n' % self.DebugFormatBool(self.for_cas_)
        return res