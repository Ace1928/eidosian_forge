from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class ContinueQueryRequest(ProtocolBuffer.ProtocolMessage):
    has_query_handle_ = 0
    query_handle_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def query_handle(self):
        return self.query_handle_

    def set_query_handle(self, x):
        self.has_query_handle_ = 1
        self.query_handle_ = x

    def clear_query_handle(self):
        if self.has_query_handle_:
            self.has_query_handle_ = 0
            self.query_handle_ = ''

    def has_query_handle(self):
        return self.has_query_handle_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_query_handle():
            self.set_query_handle(x.query_handle())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_query_handle_ != x.has_query_handle_:
            return 0
        if self.has_query_handle_ and self.query_handle_ != x.query_handle_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_query_handle_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: query_handle not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.query_handle_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_query_handle_:
            n += 1
            n += self.lengthString(len(self.query_handle_))
        return n

    def Clear(self):
        self.clear_query_handle()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.query_handle_)

    def OutputPartial(self, out):
        if self.has_query_handle_:
            out.putVarInt32(10)
            out.putPrefixedString(self.query_handle_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_query_handle(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_query_handle_:
            res += prefix + 'query_handle: %s\n' % self.DebugFormatString(self.query_handle_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kquery_handle = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'query_handle'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.ContinueQueryRequest'