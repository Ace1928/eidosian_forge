from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueFetchQueuesRequest(ProtocolBuffer.ProtocolMessage):
    has_app_id_ = 0
    app_id_ = ''
    has_max_rows_ = 0
    max_rows_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def app_id(self):
        return self.app_id_

    def set_app_id(self, x):
        self.has_app_id_ = 1
        self.app_id_ = x

    def clear_app_id(self):
        if self.has_app_id_:
            self.has_app_id_ = 0
            self.app_id_ = ''

    def has_app_id(self):
        return self.has_app_id_

    def max_rows(self):
        return self.max_rows_

    def set_max_rows(self, x):
        self.has_max_rows_ = 1
        self.max_rows_ = x

    def clear_max_rows(self):
        if self.has_max_rows_:
            self.has_max_rows_ = 0
            self.max_rows_ = 0

    def has_max_rows(self):
        return self.has_max_rows_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_max_rows():
            self.set_max_rows(x.max_rows())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
        if self.has_max_rows_ != x.has_max_rows_:
            return 0
        if self.has_max_rows_ and self.max_rows_ != x.max_rows_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_max_rows_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: max_rows not set.')
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += self.lengthVarInt64(self.max_rows_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_max_rows_:
            n += 1
            n += self.lengthVarInt64(self.max_rows_)
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_max_rows()

    def OutputUnchecked(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        out.putVarInt32(16)
        out.putVarInt32(self.max_rows_)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        if self.has_max_rows_:
            out.putVarInt32(16)
            out.putVarInt32(self.max_rows_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 16:
                self.set_max_rows(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        if self.has_max_rows_:
            res += prefix + 'max_rows: %s\n' % self.DebugFormatInt32(self.max_rows_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kmax_rows = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'max_rows'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueFetchQueuesRequest'