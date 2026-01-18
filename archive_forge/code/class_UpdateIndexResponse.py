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
class UpdateIndexResponse(ProtocolBuffer.ProtocolMessage):
    has_type_url_ = 0
    type_url_ = ''
    has_value_ = 0
    value_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def type_url(self):
        return self.type_url_

    def set_type_url(self, x):
        self.has_type_url_ = 1
        self.type_url_ = x

    def clear_type_url(self):
        if self.has_type_url_:
            self.has_type_url_ = 0
            self.type_url_ = ''

    def has_type_url(self):
        return self.has_type_url_

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

    def MergeFrom(self, x):
        assert x is not self
        if x.has_type_url():
            self.set_type_url(x.type_url())
        if x.has_value():
            self.set_value(x.value())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_type_url_ != x.has_type_url_:
            return 0
        if self.has_type_url_ and self.type_url_ != x.type_url_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_type_url_:
            n += 1 + self.lengthString(len(self.type_url_))
        if self.has_value_:
            n += 1 + self.lengthString(len(self.value_))
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_type_url_:
            n += 1 + self.lengthString(len(self.type_url_))
        if self.has_value_:
            n += 1 + self.lengthString(len(self.value_))
        return n

    def Clear(self):
        self.clear_type_url()
        self.clear_value()

    def OutputUnchecked(self, out):
        if self.has_type_url_:
            out.putVarInt32(10)
            out.putPrefixedString(self.type_url_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putPrefixedString(self.value_)

    def OutputPartial(self, out):
        if self.has_type_url_:
            out.putVarInt32(10)
            out.putPrefixedString(self.type_url_)
        if self.has_value_:
            out.putVarInt32(18)
            out.putPrefixedString(self.value_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_type_url(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_value(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_type_url_:
            res += prefix + 'type_url: %s\n' % self.DebugFormatString(self.type_url_)
        if self.has_value_:
            res += prefix + 'value: %s\n' % self.DebugFormatString(self.value_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    ktype_url = 1
    kvalue = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'type_url', 2: 'value'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.UpdateIndexResponse'