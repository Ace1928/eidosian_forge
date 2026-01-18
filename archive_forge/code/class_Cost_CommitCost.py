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
class Cost_CommitCost(ProtocolBuffer.ProtocolMessage):
    has_requested_entity_puts_ = 0
    requested_entity_puts_ = 0
    has_requested_entity_deletes_ = 0
    requested_entity_deletes_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def requested_entity_puts(self):
        return self.requested_entity_puts_

    def set_requested_entity_puts(self, x):
        self.has_requested_entity_puts_ = 1
        self.requested_entity_puts_ = x

    def clear_requested_entity_puts(self):
        if self.has_requested_entity_puts_:
            self.has_requested_entity_puts_ = 0
            self.requested_entity_puts_ = 0

    def has_requested_entity_puts(self):
        return self.has_requested_entity_puts_

    def requested_entity_deletes(self):
        return self.requested_entity_deletes_

    def set_requested_entity_deletes(self, x):
        self.has_requested_entity_deletes_ = 1
        self.requested_entity_deletes_ = x

    def clear_requested_entity_deletes(self):
        if self.has_requested_entity_deletes_:
            self.has_requested_entity_deletes_ = 0
            self.requested_entity_deletes_ = 0

    def has_requested_entity_deletes(self):
        return self.has_requested_entity_deletes_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_requested_entity_puts():
            self.set_requested_entity_puts(x.requested_entity_puts())
        if x.has_requested_entity_deletes():
            self.set_requested_entity_deletes(x.requested_entity_deletes())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_requested_entity_puts_ != x.has_requested_entity_puts_:
            return 0
        if self.has_requested_entity_puts_ and self.requested_entity_puts_ != x.requested_entity_puts_:
            return 0
        if self.has_requested_entity_deletes_ != x.has_requested_entity_deletes_:
            return 0
        if self.has_requested_entity_deletes_ and self.requested_entity_deletes_ != x.requested_entity_deletes_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_requested_entity_puts_:
            n += 1 + self.lengthVarInt64(self.requested_entity_puts_)
        if self.has_requested_entity_deletes_:
            n += 1 + self.lengthVarInt64(self.requested_entity_deletes_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_requested_entity_puts_:
            n += 1 + self.lengthVarInt64(self.requested_entity_puts_)
        if self.has_requested_entity_deletes_:
            n += 1 + self.lengthVarInt64(self.requested_entity_deletes_)
        return n

    def Clear(self):
        self.clear_requested_entity_puts()
        self.clear_requested_entity_deletes()

    def OutputUnchecked(self, out):
        if self.has_requested_entity_puts_:
            out.putVarInt32(48)
            out.putVarInt32(self.requested_entity_puts_)
        if self.has_requested_entity_deletes_:
            out.putVarInt32(56)
            out.putVarInt32(self.requested_entity_deletes_)

    def OutputPartial(self, out):
        if self.has_requested_entity_puts_:
            out.putVarInt32(48)
            out.putVarInt32(self.requested_entity_puts_)
        if self.has_requested_entity_deletes_:
            out.putVarInt32(56)
            out.putVarInt32(self.requested_entity_deletes_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 44:
                break
            if tt == 48:
                self.set_requested_entity_puts(d.getVarInt32())
                continue
            if tt == 56:
                self.set_requested_entity_deletes(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_requested_entity_puts_:
            res += prefix + 'requested_entity_puts: %s\n' % self.DebugFormatInt32(self.requested_entity_puts_)
        if self.has_requested_entity_deletes_:
            res += prefix + 'requested_entity_deletes: %s\n' % self.DebugFormatInt32(self.requested_entity_deletes_)
        return res