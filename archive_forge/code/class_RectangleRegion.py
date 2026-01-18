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
class RectangleRegion(ProtocolBuffer.ProtocolMessage):
    has_southwest_ = 0
    has_northeast_ = 0

    def __init__(self, contents=None):
        self.southwest_ = RegionPoint()
        self.northeast_ = RegionPoint()
        if contents is not None:
            self.MergeFromString(contents)

    def southwest(self):
        return self.southwest_

    def mutable_southwest(self):
        self.has_southwest_ = 1
        return self.southwest_

    def clear_southwest(self):
        self.has_southwest_ = 0
        self.southwest_.Clear()

    def has_southwest(self):
        return self.has_southwest_

    def northeast(self):
        return self.northeast_

    def mutable_northeast(self):
        self.has_northeast_ = 1
        return self.northeast_

    def clear_northeast(self):
        self.has_northeast_ = 0
        self.northeast_.Clear()

    def has_northeast(self):
        return self.has_northeast_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_southwest():
            self.mutable_southwest().MergeFrom(x.southwest())
        if x.has_northeast():
            self.mutable_northeast().MergeFrom(x.northeast())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_southwest_ != x.has_southwest_:
            return 0
        if self.has_southwest_ and self.southwest_ != x.southwest_:
            return 0
        if self.has_northeast_ != x.has_northeast_:
            return 0
        if self.has_northeast_ and self.northeast_ != x.northeast_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_southwest_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: southwest not set.')
        elif not self.southwest_.IsInitialized(debug_strs):
            initialized = 0
        if not self.has_northeast_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: northeast not set.')
        elif not self.northeast_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.southwest_.ByteSize())
        n += self.lengthString(self.northeast_.ByteSize())
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_southwest_:
            n += 1
            n += self.lengthString(self.southwest_.ByteSizePartial())
        if self.has_northeast_:
            n += 1
            n += self.lengthString(self.northeast_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_southwest()
        self.clear_northeast()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.southwest_.ByteSize())
        self.southwest_.OutputUnchecked(out)
        out.putVarInt32(18)
        out.putVarInt32(self.northeast_.ByteSize())
        self.northeast_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_southwest_:
            out.putVarInt32(10)
            out.putVarInt32(self.southwest_.ByteSizePartial())
            self.southwest_.OutputPartial(out)
        if self.has_northeast_:
            out.putVarInt32(18)
            out.putVarInt32(self.northeast_.ByteSizePartial())
            self.northeast_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_southwest().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_northeast().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_southwest_:
            res += prefix + 'southwest <\n'
            res += self.southwest_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_northeast_:
            res += prefix + 'northeast <\n'
            res += self.northeast_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    ksouthwest = 1
    knortheast = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'southwest', 2: 'northeast'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.RectangleRegion'