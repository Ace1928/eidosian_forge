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
class GeoRegion(ProtocolBuffer.ProtocolMessage):
    has_circle_ = 0
    circle_ = None
    has_rectangle_ = 0
    rectangle_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def circle(self):
        if self.circle_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.circle_ is None:
                    self.circle_ = CircleRegion()
            finally:
                self.lazy_init_lock_.release()
        return self.circle_

    def mutable_circle(self):
        self.has_circle_ = 1
        return self.circle()

    def clear_circle(self):
        if self.has_circle_:
            self.has_circle_ = 0
            if self.circle_ is not None:
                self.circle_.Clear()

    def has_circle(self):
        return self.has_circle_

    def rectangle(self):
        if self.rectangle_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.rectangle_ is None:
                    self.rectangle_ = RectangleRegion()
            finally:
                self.lazy_init_lock_.release()
        return self.rectangle_

    def mutable_rectangle(self):
        self.has_rectangle_ = 1
        return self.rectangle()

    def clear_rectangle(self):
        if self.has_rectangle_:
            self.has_rectangle_ = 0
            if self.rectangle_ is not None:
                self.rectangle_.Clear()

    def has_rectangle(self):
        return self.has_rectangle_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_circle():
            self.mutable_circle().MergeFrom(x.circle())
        if x.has_rectangle():
            self.mutable_rectangle().MergeFrom(x.rectangle())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_circle_ != x.has_circle_:
            return 0
        if self.has_circle_ and self.circle_ != x.circle_:
            return 0
        if self.has_rectangle_ != x.has_rectangle_:
            return 0
        if self.has_rectangle_ and self.rectangle_ != x.rectangle_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_circle_ and (not self.circle_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_rectangle_ and (not self.rectangle_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_circle_:
            n += 1 + self.lengthString(self.circle_.ByteSize())
        if self.has_rectangle_:
            n += 1 + self.lengthString(self.rectangle_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_circle_:
            n += 1 + self.lengthString(self.circle_.ByteSizePartial())
        if self.has_rectangle_:
            n += 1 + self.lengthString(self.rectangle_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_circle()
        self.clear_rectangle()

    def OutputUnchecked(self, out):
        if self.has_circle_:
            out.putVarInt32(10)
            out.putVarInt32(self.circle_.ByteSize())
            self.circle_.OutputUnchecked(out)
        if self.has_rectangle_:
            out.putVarInt32(18)
            out.putVarInt32(self.rectangle_.ByteSize())
            self.rectangle_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_circle_:
            out.putVarInt32(10)
            out.putVarInt32(self.circle_.ByteSizePartial())
            self.circle_.OutputPartial(out)
        if self.has_rectangle_:
            out.putVarInt32(18)
            out.putVarInt32(self.rectangle_.ByteSizePartial())
            self.rectangle_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_circle().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_rectangle().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_circle_:
            res += prefix + 'circle <\n'
            res += self.circle_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_rectangle_:
            res += prefix + 'rectangle <\n'
            res += self.rectangle_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kcircle = 1
    krectangle = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'circle', 2: 'rectangle'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.GeoRegion'