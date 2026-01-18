from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class GeoPoint(ProtocolBuffer.ProtocolMessage):
    has_latitude_ = 0
    latitude_ = 0.0
    has_longitude_ = 0
    longitude_ = 0.0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def latitude(self):
        return self.latitude_

    def set_latitude(self, x):
        self.has_latitude_ = 1
        self.latitude_ = x

    def clear_latitude(self):
        if self.has_latitude_:
            self.has_latitude_ = 0
            self.latitude_ = 0.0

    def has_latitude(self):
        return self.has_latitude_

    def longitude(self):
        return self.longitude_

    def set_longitude(self, x):
        self.has_longitude_ = 1
        self.longitude_ = x

    def clear_longitude(self):
        if self.has_longitude_:
            self.has_longitude_ = 0
            self.longitude_ = 0.0

    def has_longitude(self):
        return self.has_longitude_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_latitude():
            self.set_latitude(x.latitude())
        if x.has_longitude():
            self.set_longitude(x.longitude())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_latitude_ != x.has_latitude_:
            return 0
        if self.has_latitude_ and self.latitude_ != x.latitude_:
            return 0
        if self.has_longitude_ != x.has_longitude_:
            return 0
        if self.has_longitude_ and self.longitude_ != x.longitude_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_latitude_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: latitude not set.')
        if not self.has_longitude_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: longitude not set.')
        return initialized

    def ByteSize(self):
        n = 0
        return n + 18

    def ByteSizePartial(self):
        n = 0
        if self.has_latitude_:
            n += 9
        if self.has_longitude_:
            n += 9
        return n

    def Clear(self):
        self.clear_latitude()
        self.clear_longitude()

    def OutputUnchecked(self, out):
        out.putVarInt32(9)
        out.putDouble(self.latitude_)
        out.putVarInt32(17)
        out.putDouble(self.longitude_)

    def OutputPartial(self, out):
        if self.has_latitude_:
            out.putVarInt32(9)
            out.putDouble(self.latitude_)
        if self.has_longitude_:
            out.putVarInt32(17)
            out.putDouble(self.longitude_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 9:
                self.set_latitude(d.getDouble())
                continue
            if tt == 17:
                self.set_longitude(d.getDouble())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_latitude_:
            res += prefix + 'latitude: %s\n' % self.DebugFormat(self.latitude_)
        if self.has_longitude_:
            res += prefix + 'longitude: %s\n' % self.DebugFormat(self.longitude_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    klatitude = 1
    klongitude = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'latitude', 2: 'longitude'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.DOUBLE, 2: ProtocolBuffer.Encoder.DOUBLE}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.GeoPoint'