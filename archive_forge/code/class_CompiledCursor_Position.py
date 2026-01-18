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
class CompiledCursor_Position(ProtocolBuffer.ProtocolMessage):
    has_start_key_ = 0
    start_key_ = ''
    has_key_ = 0
    key_ = None
    has_start_inclusive_ = 0
    start_inclusive_ = 1
    has_before_ascending_ = 0
    before_ascending_ = 0

    def __init__(self, contents=None):
        self.indexvalue_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

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

    def indexvalue_size(self):
        return len(self.indexvalue_)

    def indexvalue_list(self):
        return self.indexvalue_

    def indexvalue(self, i):
        return self.indexvalue_[i]

    def mutable_indexvalue(self, i):
        return self.indexvalue_[i]

    def add_indexvalue(self):
        x = CompiledCursor_PositionIndexValue()
        self.indexvalue_.append(x)
        return x

    def clear_indexvalue(self):
        self.indexvalue_ = []

    def key(self):
        if self.key_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.key_ is None:
                    self.key_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
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

    def start_inclusive(self):
        return self.start_inclusive_

    def set_start_inclusive(self, x):
        self.has_start_inclusive_ = 1
        self.start_inclusive_ = x

    def clear_start_inclusive(self):
        if self.has_start_inclusive_:
            self.has_start_inclusive_ = 0
            self.start_inclusive_ = 1

    def has_start_inclusive(self):
        return self.has_start_inclusive_

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
        if x.has_start_key():
            self.set_start_key(x.start_key())
        for i in range(x.indexvalue_size()):
            self.add_indexvalue().CopyFrom(x.indexvalue(i))
        if x.has_key():
            self.mutable_key().MergeFrom(x.key())
        if x.has_start_inclusive():
            self.set_start_inclusive(x.start_inclusive())
        if x.has_before_ascending():
            self.set_before_ascending(x.before_ascending())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_start_key_ != x.has_start_key_:
            return 0
        if self.has_start_key_ and self.start_key_ != x.start_key_:
            return 0
        if len(self.indexvalue_) != len(x.indexvalue_):
            return 0
        for e1, e2 in zip(self.indexvalue_, x.indexvalue_):
            if e1 != e2:
                return 0
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_start_inclusive_ != x.has_start_inclusive_:
            return 0
        if self.has_start_inclusive_ and self.start_inclusive_ != x.start_inclusive_:
            return 0
        if self.has_before_ascending_ != x.has_before_ascending_:
            return 0
        if self.has_before_ascending_ and self.before_ascending_ != x.before_ascending_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.indexvalue_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_key_ and (not self.key_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_start_key_:
            n += 2 + self.lengthString(len(self.start_key_))
        n += 4 * len(self.indexvalue_)
        for i in range(len(self.indexvalue_)):
            n += self.indexvalue_[i].ByteSize()
        if self.has_key_:
            n += 2 + self.lengthString(self.key_.ByteSize())
        if self.has_start_inclusive_:
            n += 3
        if self.has_before_ascending_:
            n += 3
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_start_key_:
            n += 2 + self.lengthString(len(self.start_key_))
        n += 4 * len(self.indexvalue_)
        for i in range(len(self.indexvalue_)):
            n += self.indexvalue_[i].ByteSizePartial()
        if self.has_key_:
            n += 2 + self.lengthString(self.key_.ByteSizePartial())
        if self.has_start_inclusive_:
            n += 3
        if self.has_before_ascending_:
            n += 3
        return n

    def Clear(self):
        self.clear_start_key()
        self.clear_indexvalue()
        self.clear_key()
        self.clear_start_inclusive()
        self.clear_before_ascending()

    def OutputUnchecked(self, out):
        if self.has_start_key_:
            out.putVarInt32(218)
            out.putPrefixedString(self.start_key_)
        if self.has_start_inclusive_:
            out.putVarInt32(224)
            out.putBoolean(self.start_inclusive_)
        for i in range(len(self.indexvalue_)):
            out.putVarInt32(235)
            self.indexvalue_[i].OutputUnchecked(out)
            out.putVarInt32(236)
        if self.has_key_:
            out.putVarInt32(258)
            out.putVarInt32(self.key_.ByteSize())
            self.key_.OutputUnchecked(out)
        if self.has_before_ascending_:
            out.putVarInt32(264)
            out.putBoolean(self.before_ascending_)

    def OutputPartial(self, out):
        if self.has_start_key_:
            out.putVarInt32(218)
            out.putPrefixedString(self.start_key_)
        if self.has_start_inclusive_:
            out.putVarInt32(224)
            out.putBoolean(self.start_inclusive_)
        for i in range(len(self.indexvalue_)):
            out.putVarInt32(235)
            self.indexvalue_[i].OutputPartial(out)
            out.putVarInt32(236)
        if self.has_key_:
            out.putVarInt32(258)
            out.putVarInt32(self.key_.ByteSizePartial())
            self.key_.OutputPartial(out)
        if self.has_before_ascending_:
            out.putVarInt32(264)
            out.putBoolean(self.before_ascending_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 20:
                break
            if tt == 218:
                self.set_start_key(d.getPrefixedString())
                continue
            if tt == 224:
                self.set_start_inclusive(d.getBoolean())
                continue
            if tt == 235:
                self.add_indexvalue().TryMerge(d)
                continue
            if tt == 258:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_key().TryMerge(tmp)
                continue
            if tt == 264:
                self.set_before_ascending(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_start_key_:
            res += prefix + 'start_key: %s\n' % self.DebugFormatString(self.start_key_)
        cnt = 0
        for e in self.indexvalue_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'IndexValue%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        if self.has_key_:
            res += prefix + 'key <\n'
            res += self.key_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_start_inclusive_:
            res += prefix + 'start_inclusive: %s\n' % self.DebugFormatBool(self.start_inclusive_)
        if self.has_before_ascending_:
            res += prefix + 'before_ascending: %s\n' % self.DebugFormatBool(self.before_ascending_)
        return res