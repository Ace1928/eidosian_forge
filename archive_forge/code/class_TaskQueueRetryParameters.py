from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueRetryParameters(ProtocolBuffer.ProtocolMessage):
    has_retry_limit_ = 0
    retry_limit_ = 0
    has_age_limit_sec_ = 0
    age_limit_sec_ = 0
    has_min_backoff_sec_ = 0
    min_backoff_sec_ = 0.1
    has_max_backoff_sec_ = 0
    max_backoff_sec_ = 3600.0
    has_max_doublings_ = 0
    max_doublings_ = 16

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def retry_limit(self):
        return self.retry_limit_

    def set_retry_limit(self, x):
        self.has_retry_limit_ = 1
        self.retry_limit_ = x

    def clear_retry_limit(self):
        if self.has_retry_limit_:
            self.has_retry_limit_ = 0
            self.retry_limit_ = 0

    def has_retry_limit(self):
        return self.has_retry_limit_

    def age_limit_sec(self):
        return self.age_limit_sec_

    def set_age_limit_sec(self, x):
        self.has_age_limit_sec_ = 1
        self.age_limit_sec_ = x

    def clear_age_limit_sec(self):
        if self.has_age_limit_sec_:
            self.has_age_limit_sec_ = 0
            self.age_limit_sec_ = 0

    def has_age_limit_sec(self):
        return self.has_age_limit_sec_

    def min_backoff_sec(self):
        return self.min_backoff_sec_

    def set_min_backoff_sec(self, x):
        self.has_min_backoff_sec_ = 1
        self.min_backoff_sec_ = x

    def clear_min_backoff_sec(self):
        if self.has_min_backoff_sec_:
            self.has_min_backoff_sec_ = 0
            self.min_backoff_sec_ = 0.1

    def has_min_backoff_sec(self):
        return self.has_min_backoff_sec_

    def max_backoff_sec(self):
        return self.max_backoff_sec_

    def set_max_backoff_sec(self, x):
        self.has_max_backoff_sec_ = 1
        self.max_backoff_sec_ = x

    def clear_max_backoff_sec(self):
        if self.has_max_backoff_sec_:
            self.has_max_backoff_sec_ = 0
            self.max_backoff_sec_ = 3600.0

    def has_max_backoff_sec(self):
        return self.has_max_backoff_sec_

    def max_doublings(self):
        return self.max_doublings_

    def set_max_doublings(self, x):
        self.has_max_doublings_ = 1
        self.max_doublings_ = x

    def clear_max_doublings(self):
        if self.has_max_doublings_:
            self.has_max_doublings_ = 0
            self.max_doublings_ = 16

    def has_max_doublings(self):
        return self.has_max_doublings_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_retry_limit():
            self.set_retry_limit(x.retry_limit())
        if x.has_age_limit_sec():
            self.set_age_limit_sec(x.age_limit_sec())
        if x.has_min_backoff_sec():
            self.set_min_backoff_sec(x.min_backoff_sec())
        if x.has_max_backoff_sec():
            self.set_max_backoff_sec(x.max_backoff_sec())
        if x.has_max_doublings():
            self.set_max_doublings(x.max_doublings())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_retry_limit_ != x.has_retry_limit_:
            return 0
        if self.has_retry_limit_ and self.retry_limit_ != x.retry_limit_:
            return 0
        if self.has_age_limit_sec_ != x.has_age_limit_sec_:
            return 0
        if self.has_age_limit_sec_ and self.age_limit_sec_ != x.age_limit_sec_:
            return 0
        if self.has_min_backoff_sec_ != x.has_min_backoff_sec_:
            return 0
        if self.has_min_backoff_sec_ and self.min_backoff_sec_ != x.min_backoff_sec_:
            return 0
        if self.has_max_backoff_sec_ != x.has_max_backoff_sec_:
            return 0
        if self.has_max_backoff_sec_ and self.max_backoff_sec_ != x.max_backoff_sec_:
            return 0
        if self.has_max_doublings_ != x.has_max_doublings_:
            return 0
        if self.has_max_doublings_ and self.max_doublings_ != x.max_doublings_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_retry_limit_:
            n += 1 + self.lengthVarInt64(self.retry_limit_)
        if self.has_age_limit_sec_:
            n += 1 + self.lengthVarInt64(self.age_limit_sec_)
        if self.has_min_backoff_sec_:
            n += 9
        if self.has_max_backoff_sec_:
            n += 9
        if self.has_max_doublings_:
            n += 1 + self.lengthVarInt64(self.max_doublings_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_retry_limit_:
            n += 1 + self.lengthVarInt64(self.retry_limit_)
        if self.has_age_limit_sec_:
            n += 1 + self.lengthVarInt64(self.age_limit_sec_)
        if self.has_min_backoff_sec_:
            n += 9
        if self.has_max_backoff_sec_:
            n += 9
        if self.has_max_doublings_:
            n += 1 + self.lengthVarInt64(self.max_doublings_)
        return n

    def Clear(self):
        self.clear_retry_limit()
        self.clear_age_limit_sec()
        self.clear_min_backoff_sec()
        self.clear_max_backoff_sec()
        self.clear_max_doublings()

    def OutputUnchecked(self, out):
        if self.has_retry_limit_:
            out.putVarInt32(8)
            out.putVarInt32(self.retry_limit_)
        if self.has_age_limit_sec_:
            out.putVarInt32(16)
            out.putVarInt64(self.age_limit_sec_)
        if self.has_min_backoff_sec_:
            out.putVarInt32(25)
            out.putDouble(self.min_backoff_sec_)
        if self.has_max_backoff_sec_:
            out.putVarInt32(33)
            out.putDouble(self.max_backoff_sec_)
        if self.has_max_doublings_:
            out.putVarInt32(40)
            out.putVarInt32(self.max_doublings_)

    def OutputPartial(self, out):
        if self.has_retry_limit_:
            out.putVarInt32(8)
            out.putVarInt32(self.retry_limit_)
        if self.has_age_limit_sec_:
            out.putVarInt32(16)
            out.putVarInt64(self.age_limit_sec_)
        if self.has_min_backoff_sec_:
            out.putVarInt32(25)
            out.putDouble(self.min_backoff_sec_)
        if self.has_max_backoff_sec_:
            out.putVarInt32(33)
            out.putDouble(self.max_backoff_sec_)
        if self.has_max_doublings_:
            out.putVarInt32(40)
            out.putVarInt32(self.max_doublings_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_retry_limit(d.getVarInt32())
                continue
            if tt == 16:
                self.set_age_limit_sec(d.getVarInt64())
                continue
            if tt == 25:
                self.set_min_backoff_sec(d.getDouble())
                continue
            if tt == 33:
                self.set_max_backoff_sec(d.getDouble())
                continue
            if tt == 40:
                self.set_max_doublings(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_retry_limit_:
            res += prefix + 'retry_limit: %s\n' % self.DebugFormatInt32(self.retry_limit_)
        if self.has_age_limit_sec_:
            res += prefix + 'age_limit_sec: %s\n' % self.DebugFormatInt64(self.age_limit_sec_)
        if self.has_min_backoff_sec_:
            res += prefix + 'min_backoff_sec: %s\n' % self.DebugFormat(self.min_backoff_sec_)
        if self.has_max_backoff_sec_:
            res += prefix + 'max_backoff_sec: %s\n' % self.DebugFormat(self.max_backoff_sec_)
        if self.has_max_doublings_:
            res += prefix + 'max_doublings: %s\n' % self.DebugFormatInt32(self.max_doublings_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kretry_limit = 1
    kage_limit_sec = 2
    kmin_backoff_sec = 3
    kmax_backoff_sec = 4
    kmax_doublings = 5
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'retry_limit', 2: 'age_limit_sec', 3: 'min_backoff_sec', 4: 'max_backoff_sec', 5: 'max_doublings'}, 5)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.DOUBLE, 4: ProtocolBuffer.Encoder.DOUBLE, 5: ProtocolBuffer.Encoder.NUMERIC}, 5, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueRetryParameters'