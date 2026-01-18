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
class CompiledQuery_MergeJoinScan(ProtocolBuffer.ProtocolMessage):
    has_index_name_ = 0
    index_name_ = ''
    has_value_prefix_ = 0
    value_prefix_ = 0

    def __init__(self, contents=None):
        self.prefix_value_ = []
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

    def prefix_value_size(self):
        return len(self.prefix_value_)

    def prefix_value_list(self):
        return self.prefix_value_

    def prefix_value(self, i):
        return self.prefix_value_[i]

    def set_prefix_value(self, i, x):
        self.prefix_value_[i] = x

    def add_prefix_value(self, x):
        self.prefix_value_.append(x)

    def clear_prefix_value(self):
        self.prefix_value_ = []

    def value_prefix(self):
        return self.value_prefix_

    def set_value_prefix(self, x):
        self.has_value_prefix_ = 1
        self.value_prefix_ = x

    def clear_value_prefix(self):
        if self.has_value_prefix_:
            self.has_value_prefix_ = 0
            self.value_prefix_ = 0

    def has_value_prefix(self):
        return self.has_value_prefix_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_name():
            self.set_index_name(x.index_name())
        for i in range(x.prefix_value_size()):
            self.add_prefix_value(x.prefix_value(i))
        if x.has_value_prefix():
            self.set_value_prefix(x.value_prefix())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_name_ != x.has_index_name_:
            return 0
        if self.has_index_name_ and self.index_name_ != x.index_name_:
            return 0
        if len(self.prefix_value_) != len(x.prefix_value_):
            return 0
        for e1, e2 in zip(self.prefix_value_, x.prefix_value_):
            if e1 != e2:
                return 0
        if self.has_value_prefix_ != x.has_value_prefix_:
            return 0
        if self.has_value_prefix_ and self.value_prefix_ != x.value_prefix_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_index_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: index_name not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.index_name_))
        n += 1 * len(self.prefix_value_)
        for i in range(len(self.prefix_value_)):
            n += self.lengthString(len(self.prefix_value_[i]))
        if self.has_value_prefix_:
            n += 3
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_index_name_:
            n += 1
            n += self.lengthString(len(self.index_name_))
        n += 1 * len(self.prefix_value_)
        for i in range(len(self.prefix_value_)):
            n += self.lengthString(len(self.prefix_value_[i]))
        if self.has_value_prefix_:
            n += 3
        return n

    def Clear(self):
        self.clear_index_name()
        self.clear_prefix_value()
        self.clear_value_prefix()

    def OutputUnchecked(self, out):
        out.putVarInt32(66)
        out.putPrefixedString(self.index_name_)
        for i in range(len(self.prefix_value_)):
            out.putVarInt32(74)
            out.putPrefixedString(self.prefix_value_[i])
        if self.has_value_prefix_:
            out.putVarInt32(160)
            out.putBoolean(self.value_prefix_)

    def OutputPartial(self, out):
        if self.has_index_name_:
            out.putVarInt32(66)
            out.putPrefixedString(self.index_name_)
        for i in range(len(self.prefix_value_)):
            out.putVarInt32(74)
            out.putPrefixedString(self.prefix_value_[i])
        if self.has_value_prefix_:
            out.putVarInt32(160)
            out.putBoolean(self.value_prefix_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 60:
                break
            if tt == 66:
                self.set_index_name(d.getPrefixedString())
                continue
            if tt == 74:
                self.add_prefix_value(d.getPrefixedString())
                continue
            if tt == 160:
                self.set_value_prefix(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_name_:
            res += prefix + 'index_name: %s\n' % self.DebugFormatString(self.index_name_)
        cnt = 0
        for e in self.prefix_value_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'prefix_value%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_value_prefix_:
            res += prefix + 'value_prefix: %s\n' % self.DebugFormatBool(self.value_prefix_)
        return res