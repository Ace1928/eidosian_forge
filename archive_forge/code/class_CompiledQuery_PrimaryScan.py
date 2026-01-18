from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
class CompiledQuery_PrimaryScan(ProtocolBuffer.ProtocolMessage):
    has_index_name_ = 0
    index_name_ = ''
    has_start_key_ = 0
    start_key_ = ''
    has_start_inclusive_ = 0
    start_inclusive_ = 0
    has_end_key_ = 0
    end_key_ = ''
    has_end_inclusive_ = 0
    end_inclusive_ = 0
    has_end_unapplied_log_timestamp_us_ = 0
    end_unapplied_log_timestamp_us_ = 0

    def __init__(self, contents=None):
        self.start_postfix_value_ = []
        self.end_postfix_value_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def index_name(self):
        return self.index_name_

    def set_index_name(self, x):
        self.has_index_name_ = 1
        self.index_name_ = x

    def clear_index_name(self):
        if self.has_index_name_:
            self.has_index_name_ = 0
            self.index_name_ = ''

    def has_index_name(self):
        return self.has_index_name_

    def start_key(self):
        return self.start_key_

    def set_start_key(self, x):
        self.has_start_key_ = 1
        self.start_key_ = x

    def clear_start_key(self):
        if self.has_start_key_:
            self.has_start_key_ = 0
            self.start_key_ = ''

    def has_start_key(self):
        return self.has_start_key_

    def start_inclusive(self):
        return self.start_inclusive_

    def set_start_inclusive(self, x):
        self.has_start_inclusive_ = 1
        self.start_inclusive_ = x

    def clear_start_inclusive(self):
        if self.has_start_inclusive_:
            self.has_start_inclusive_ = 0
            self.start_inclusive_ = 0

    def has_start_inclusive(self):
        return self.has_start_inclusive_

    def end_key(self):
        return self.end_key_

    def set_end_key(self, x):
        self.has_end_key_ = 1
        self.end_key_ = x

    def clear_end_key(self):
        if self.has_end_key_:
            self.has_end_key_ = 0
            self.end_key_ = ''

    def has_end_key(self):
        return self.has_end_key_

    def end_inclusive(self):
        return self.end_inclusive_

    def set_end_inclusive(self, x):
        self.has_end_inclusive_ = 1
        self.end_inclusive_ = x

    def clear_end_inclusive(self):
        if self.has_end_inclusive_:
            self.has_end_inclusive_ = 0
            self.end_inclusive_ = 0

    def has_end_inclusive(self):
        return self.has_end_inclusive_

    def start_postfix_value_size(self):
        return len(self.start_postfix_value_)

    def start_postfix_value_list(self):
        return self.start_postfix_value_

    def start_postfix_value(self, i):
        return self.start_postfix_value_[i]

    def set_start_postfix_value(self, i, x):
        self.start_postfix_value_[i] = x

    def add_start_postfix_value(self, x):
        self.start_postfix_value_.append(x)

    def clear_start_postfix_value(self):
        self.start_postfix_value_ = []

    def end_postfix_value_size(self):
        return len(self.end_postfix_value_)

    def end_postfix_value_list(self):
        return self.end_postfix_value_

    def end_postfix_value(self, i):
        return self.end_postfix_value_[i]

    def set_end_postfix_value(self, i, x):
        self.end_postfix_value_[i] = x

    def add_end_postfix_value(self, x):
        self.end_postfix_value_.append(x)

    def clear_end_postfix_value(self):
        self.end_postfix_value_ = []

    def end_unapplied_log_timestamp_us(self):
        return self.end_unapplied_log_timestamp_us_

    def set_end_unapplied_log_timestamp_us(self, x):
        self.has_end_unapplied_log_timestamp_us_ = 1
        self.end_unapplied_log_timestamp_us_ = x

    def clear_end_unapplied_log_timestamp_us(self):
        if self.has_end_unapplied_log_timestamp_us_:
            self.has_end_unapplied_log_timestamp_us_ = 0
            self.end_unapplied_log_timestamp_us_ = 0

    def has_end_unapplied_log_timestamp_us(self):
        return self.has_end_unapplied_log_timestamp_us_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_name():
            self.set_index_name(x.index_name())
        if x.has_start_key():
            self.set_start_key(x.start_key())
        if x.has_start_inclusive():
            self.set_start_inclusive(x.start_inclusive())
        if x.has_end_key():
            self.set_end_key(x.end_key())
        if x.has_end_inclusive():
            self.set_end_inclusive(x.end_inclusive())
        for i in range(x.start_postfix_value_size()):
            self.add_start_postfix_value(x.start_postfix_value(i))
        for i in range(x.end_postfix_value_size()):
            self.add_end_postfix_value(x.end_postfix_value(i))
        if x.has_end_unapplied_log_timestamp_us():
            self.set_end_unapplied_log_timestamp_us(x.end_unapplied_log_timestamp_us())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_name_ != x.has_index_name_:
            return 0
        if self.has_index_name_ and self.index_name_ != x.index_name_:
            return 0
        if self.has_start_key_ != x.has_start_key_:
            return 0
        if self.has_start_key_ and self.start_key_ != x.start_key_:
            return 0
        if self.has_start_inclusive_ != x.has_start_inclusive_:
            return 0
        if self.has_start_inclusive_ and self.start_inclusive_ != x.start_inclusive_:
            return 0
        if self.has_end_key_ != x.has_end_key_:
            return 0
        if self.has_end_key_ and self.end_key_ != x.end_key_:
            return 0
        if self.has_end_inclusive_ != x.has_end_inclusive_:
            return 0
        if self.has_end_inclusive_ and self.end_inclusive_ != x.end_inclusive_:
            return 0
        if len(self.start_postfix_value_) != len(x.start_postfix_value_):
            return 0
        for e1, e2 in zip(self.start_postfix_value_, x.start_postfix_value_):
            if e1 != e2:
                return 0
        if len(self.end_postfix_value_) != len(x.end_postfix_value_):
            return 0
        for e1, e2 in zip(self.end_postfix_value_, x.end_postfix_value_):
            if e1 != e2:
                return 0
        if self.has_end_unapplied_log_timestamp_us_ != x.has_end_unapplied_log_timestamp_us_:
            return 0
        if self.has_end_unapplied_log_timestamp_us_ and self.end_unapplied_log_timestamp_us_ != x.end_unapplied_log_timestamp_us_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_index_name_:
            n += 1 + self.lengthString(len(self.index_name_))
        if self.has_start_key_:
            n += 1 + self.lengthString(len(self.start_key_))
        if self.has_start_inclusive_:
            n += 2
        if self.has_end_key_:
            n += 1 + self.lengthString(len(self.end_key_))
        if self.has_end_inclusive_:
            n += 2
        n += 2 * len(self.start_postfix_value_)
        for i in range(len(self.start_postfix_value_)):
            n += self.lengthString(len(self.start_postfix_value_[i]))
        n += 2 * len(self.end_postfix_value_)
        for i in range(len(self.end_postfix_value_)):
            n += self.lengthString(len(self.end_postfix_value_[i]))
        if self.has_end_unapplied_log_timestamp_us_:
            n += 2 + self.lengthVarInt64(self.end_unapplied_log_timestamp_us_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_index_name_:
            n += 1 + self.lengthString(len(self.index_name_))
        if self.has_start_key_:
            n += 1 + self.lengthString(len(self.start_key_))
        if self.has_start_inclusive_:
            n += 2
        if self.has_end_key_:
            n += 1 + self.lengthString(len(self.end_key_))
        if self.has_end_inclusive_:
            n += 2
        n += 2 * len(self.start_postfix_value_)
        for i in range(len(self.start_postfix_value_)):
            n += self.lengthString(len(self.start_postfix_value_[i]))
        n += 2 * len(self.end_postfix_value_)
        for i in range(len(self.end_postfix_value_)):
            n += self.lengthString(len(self.end_postfix_value_[i]))
        if self.has_end_unapplied_log_timestamp_us_:
            n += 2 + self.lengthVarInt64(self.end_unapplied_log_timestamp_us_)
        return n

    def Clear(self):
        self.clear_index_name()
        self.clear_start_key()
        self.clear_start_inclusive()
        self.clear_end_key()
        self.clear_end_inclusive()
        self.clear_start_postfix_value()
        self.clear_end_postfix_value()
        self.clear_end_unapplied_log_timestamp_us()

    def OutputUnchecked(self, out):
        if self.has_index_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.index_name_)
        if self.has_start_key_:
            out.putVarInt32(26)
            out.putPrefixedString(self.start_key_)
        if self.has_start_inclusive_:
            out.putVarInt32(32)
            out.putBoolean(self.start_inclusive_)
        if self.has_end_key_:
            out.putVarInt32(42)
            out.putPrefixedString(self.end_key_)
        if self.has_end_inclusive_:
            out.putVarInt32(48)
            out.putBoolean(self.end_inclusive_)
        if self.has_end_unapplied_log_timestamp_us_:
            out.putVarInt32(152)
            out.putVarInt64(self.end_unapplied_log_timestamp_us_)
        for i in range(len(self.start_postfix_value_)):
            out.putVarInt32(178)
            out.putPrefixedString(self.start_postfix_value_[i])
        for i in range(len(self.end_postfix_value_)):
            out.putVarInt32(186)
            out.putPrefixedString(self.end_postfix_value_[i])

    def OutputPartial(self, out):
        if self.has_index_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.index_name_)
        if self.has_start_key_:
            out.putVarInt32(26)
            out.putPrefixedString(self.start_key_)
        if self.has_start_inclusive_:
            out.putVarInt32(32)
            out.putBoolean(self.start_inclusive_)
        if self.has_end_key_:
            out.putVarInt32(42)
            out.putPrefixedString(self.end_key_)
        if self.has_end_inclusive_:
            out.putVarInt32(48)
            out.putBoolean(self.end_inclusive_)
        if self.has_end_unapplied_log_timestamp_us_:
            out.putVarInt32(152)
            out.putVarInt64(self.end_unapplied_log_timestamp_us_)
        for i in range(len(self.start_postfix_value_)):
            out.putVarInt32(178)
            out.putPrefixedString(self.start_postfix_value_[i])
        for i in range(len(self.end_postfix_value_)):
            out.putVarInt32(186)
            out.putPrefixedString(self.end_postfix_value_[i])

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_index_name(d.getPrefixedString())
                continue
            if tt == 26:
                self.set_start_key(d.getPrefixedString())
                continue
            if tt == 32:
                self.set_start_inclusive(d.getBoolean())
                continue
            if tt == 42:
                self.set_end_key(d.getPrefixedString())
                continue
            if tt == 48:
                self.set_end_inclusive(d.getBoolean())
                continue
            if tt == 152:
                self.set_end_unapplied_log_timestamp_us(d.getVarInt64())
                continue
            if tt == 178:
                self.add_start_postfix_value(d.getPrefixedString())
                continue
            if tt == 186:
                self.add_end_postfix_value(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_name_:
            res += prefix + 'index_name: %s\n' % self.DebugFormatString(self.index_name_)
        if self.has_start_key_:
            res += prefix + 'start_key: %s\n' % self.DebugFormatString(self.start_key_)
        if self.has_start_inclusive_:
            res += prefix + 'start_inclusive: %s\n' % self.DebugFormatBool(self.start_inclusive_)
        if self.has_end_key_:
            res += prefix + 'end_key: %s\n' % self.DebugFormatString(self.end_key_)
        if self.has_end_inclusive_:
            res += prefix + 'end_inclusive: %s\n' % self.DebugFormatBool(self.end_inclusive_)
        cnt = 0
        for e in self.start_postfix_value_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'start_postfix_value%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        cnt = 0
        for e in self.end_postfix_value_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'end_postfix_value%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_end_unapplied_log_timestamp_us_:
            res += prefix + 'end_unapplied_log_timestamp_us: %s\n' % self.DebugFormatInt64(self.end_unapplied_log_timestamp_us_)
        return res