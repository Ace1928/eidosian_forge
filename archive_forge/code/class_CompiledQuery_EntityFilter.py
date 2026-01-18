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
class CompiledQuery_EntityFilter(ProtocolBuffer.ProtocolMessage):
    has_distinct_ = 0
    distinct_ = 0
    has_kind_ = 0
    kind_ = ''
    has_ancestor_ = 0
    ancestor_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def distinct(self):
        return self.distinct_

    def set_distinct(self, x):
        self.has_distinct_ = 1
        self.distinct_ = x

    def clear_distinct(self):
        if self.has_distinct_:
            self.has_distinct_ = 0
            self.distinct_ = 0

    def has_distinct(self):
        return self.has_distinct_

    def kind(self):
        return self.kind_

    def set_kind(self, x):
        self.has_kind_ = 1
        self.kind_ = x

    def clear_kind(self):
        if self.has_kind_:
            self.has_kind_ = 0
            self.kind_ = ''

    def has_kind(self):
        return self.has_kind_

    def ancestor(self):
        if self.ancestor_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.ancestor_ is None:
                    self.ancestor_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
            finally:
                self.lazy_init_lock_.release()
        return self.ancestor_

    def mutable_ancestor(self):
        self.has_ancestor_ = 1
        return self.ancestor()

    def clear_ancestor(self):
        if self.has_ancestor_:
            self.has_ancestor_ = 0
            if self.ancestor_ is not None:
                self.ancestor_.Clear()

    def has_ancestor(self):
        return self.has_ancestor_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_distinct():
            self.set_distinct(x.distinct())
        if x.has_kind():
            self.set_kind(x.kind())
        if x.has_ancestor():
            self.mutable_ancestor().MergeFrom(x.ancestor())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_distinct_ != x.has_distinct_:
            return 0
        if self.has_distinct_ and self.distinct_ != x.distinct_:
            return 0
        if self.has_kind_ != x.has_kind_:
            return 0
        if self.has_kind_ and self.kind_ != x.kind_:
            return 0
        if self.has_ancestor_ != x.has_ancestor_:
            return 0
        if self.has_ancestor_ and self.ancestor_ != x.ancestor_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_ancestor_ and (not self.ancestor_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_distinct_:
            n += 2
        if self.has_kind_:
            n += 2 + self.lengthString(len(self.kind_))
        if self.has_ancestor_:
            n += 2 + self.lengthString(self.ancestor_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_distinct_:
            n += 2
        if self.has_kind_:
            n += 2 + self.lengthString(len(self.kind_))
        if self.has_ancestor_:
            n += 2 + self.lengthString(self.ancestor_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_distinct()
        self.clear_kind()
        self.clear_ancestor()

    def OutputUnchecked(self, out):
        if self.has_distinct_:
            out.putVarInt32(112)
            out.putBoolean(self.distinct_)
        if self.has_kind_:
            out.putVarInt32(138)
            out.putPrefixedString(self.kind_)
        if self.has_ancestor_:
            out.putVarInt32(146)
            out.putVarInt32(self.ancestor_.ByteSize())
            self.ancestor_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_distinct_:
            out.putVarInt32(112)
            out.putBoolean(self.distinct_)
        if self.has_kind_:
            out.putVarInt32(138)
            out.putPrefixedString(self.kind_)
        if self.has_ancestor_:
            out.putVarInt32(146)
            out.putVarInt32(self.ancestor_.ByteSizePartial())
            self.ancestor_.OutputPartial(out)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 108:
                break
            if tt == 112:
                self.set_distinct(d.getBoolean())
                continue
            if tt == 138:
                self.set_kind(d.getPrefixedString())
                continue
            if tt == 146:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_ancestor().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_distinct_:
            res += prefix + 'distinct: %s\n' % self.DebugFormatBool(self.distinct_)
        if self.has_kind_:
            res += prefix + 'kind: %s\n' % self.DebugFormatString(self.kind_)
        if self.has_ancestor_:
            res += prefix + 'ancestor <\n'
            res += self.ancestor_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res