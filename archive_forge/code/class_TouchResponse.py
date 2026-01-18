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
class TouchResponse(ProtocolBuffer.ProtocolMessage):
    has_cost_ = 0
    cost_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def cost(self):
        if self.cost_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.cost_ is None:
                    self.cost_ = Cost()
            finally:
                self.lazy_init_lock_.release()
        return self.cost_

    def mutable_cost(self):
        self.has_cost_ = 1
        return self.cost()

    def clear_cost(self):
        if self.has_cost_:
            self.has_cost_ = 0
            if self.cost_ is not None:
                self.cost_.Clear()

    def has_cost(self):
        return self.has_cost_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_cost():
            self.mutable_cost().MergeFrom(x.cost())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_cost_ != x.has_cost_:
            return 0
        if self.has_cost_ and self.cost_ != x.cost_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_cost_ and (not self.cost_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_cost_:
            n += 1 + self.lengthString(self.cost_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_cost_:
            n += 1 + self.lengthString(self.cost_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_cost()

    def OutputUnchecked(self, out):
        if self.has_cost_:
            out.putVarInt32(10)
            out.putVarInt32(self.cost_.ByteSize())
            self.cost_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_cost_:
            out.putVarInt32(10)
            out.putVarInt32(self.cost_.ByteSizePartial())
            self.cost_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_cost().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_cost_:
            res += prefix + 'cost <\n'
            res += self.cost_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kcost = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'cost'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.TouchResponse'