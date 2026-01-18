from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryTasksResponse_TaskRunLog(ProtocolBuffer.ProtocolMessage):
    has_dispatched_usec_ = 0
    dispatched_usec_ = 0
    has_lag_usec_ = 0
    lag_usec_ = 0
    has_elapsed_usec_ = 0
    elapsed_usec_ = 0
    has_response_code_ = 0
    response_code_ = 0
    has_retry_reason_ = 0
    retry_reason_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def dispatched_usec(self):
        return self.dispatched_usec_

    def set_dispatched_usec(self, x):
        self.has_dispatched_usec_ = 1
        self.dispatched_usec_ = x

    def clear_dispatched_usec(self):
        if self.has_dispatched_usec_:
            self.has_dispatched_usec_ = 0
            self.dispatched_usec_ = 0

    def has_dispatched_usec(self):
        return self.has_dispatched_usec_

    def lag_usec(self):
        return self.lag_usec_

    def set_lag_usec(self, x):
        self.has_lag_usec_ = 1
        self.lag_usec_ = x

    def clear_lag_usec(self):
        if self.has_lag_usec_:
            self.has_lag_usec_ = 0
            self.lag_usec_ = 0

    def has_lag_usec(self):
        return self.has_lag_usec_

    def elapsed_usec(self):
        return self.elapsed_usec_

    def set_elapsed_usec(self, x):
        self.has_elapsed_usec_ = 1
        self.elapsed_usec_ = x

    def clear_elapsed_usec(self):
        if self.has_elapsed_usec_:
            self.has_elapsed_usec_ = 0
            self.elapsed_usec_ = 0

    def has_elapsed_usec(self):
        return self.has_elapsed_usec_

    def response_code(self):
        return self.response_code_

    def set_response_code(self, x):
        self.has_response_code_ = 1
        self.response_code_ = x

    def clear_response_code(self):
        if self.has_response_code_:
            self.has_response_code_ = 0
            self.response_code_ = 0

    def has_response_code(self):
        return self.has_response_code_

    def retry_reason(self):
        return self.retry_reason_

    def set_retry_reason(self, x):
        self.has_retry_reason_ = 1
        self.retry_reason_ = x

    def clear_retry_reason(self):
        if self.has_retry_reason_:
            self.has_retry_reason_ = 0
            self.retry_reason_ = ''

    def has_retry_reason(self):
        return self.has_retry_reason_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_dispatched_usec():
            self.set_dispatched_usec(x.dispatched_usec())
        if x.has_lag_usec():
            self.set_lag_usec(x.lag_usec())
        if x.has_elapsed_usec():
            self.set_elapsed_usec(x.elapsed_usec())
        if x.has_response_code():
            self.set_response_code(x.response_code())
        if x.has_retry_reason():
            self.set_retry_reason(x.retry_reason())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_dispatched_usec_ != x.has_dispatched_usec_:
            return 0
        if self.has_dispatched_usec_ and self.dispatched_usec_ != x.dispatched_usec_:
            return 0
        if self.has_lag_usec_ != x.has_lag_usec_:
            return 0
        if self.has_lag_usec_ and self.lag_usec_ != x.lag_usec_:
            return 0
        if self.has_elapsed_usec_ != x.has_elapsed_usec_:
            return 0
        if self.has_elapsed_usec_ and self.elapsed_usec_ != x.elapsed_usec_:
            return 0
        if self.has_response_code_ != x.has_response_code_:
            return 0
        if self.has_response_code_ and self.response_code_ != x.response_code_:
            return 0
        if self.has_retry_reason_ != x.has_retry_reason_:
            return 0
        if self.has_retry_reason_ and self.retry_reason_ != x.retry_reason_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_dispatched_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: dispatched_usec not set.')
        if not self.has_lag_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: lag_usec not set.')
        if not self.has_elapsed_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: elapsed_usec not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.dispatched_usec_)
        n += self.lengthVarInt64(self.lag_usec_)
        n += self.lengthVarInt64(self.elapsed_usec_)
        if self.has_response_code_:
            n += 2 + self.lengthVarInt64(self.response_code_)
        if self.has_retry_reason_:
            n += 2 + self.lengthString(len(self.retry_reason_))
        return n + 6

    def ByteSizePartial(self):
        n = 0
        if self.has_dispatched_usec_:
            n += 2
            n += self.lengthVarInt64(self.dispatched_usec_)
        if self.has_lag_usec_:
            n += 2
            n += self.lengthVarInt64(self.lag_usec_)
        if self.has_elapsed_usec_:
            n += 2
            n += self.lengthVarInt64(self.elapsed_usec_)
        if self.has_response_code_:
            n += 2 + self.lengthVarInt64(self.response_code_)
        if self.has_retry_reason_:
            n += 2 + self.lengthString(len(self.retry_reason_))
        return n

    def Clear(self):
        self.clear_dispatched_usec()
        self.clear_lag_usec()
        self.clear_elapsed_usec()
        self.clear_response_code()
        self.clear_retry_reason()

    def OutputUnchecked(self, out):
        out.putVarInt32(136)
        out.putVarInt64(self.dispatched_usec_)
        out.putVarInt32(144)
        out.putVarInt64(self.lag_usec_)
        out.putVarInt32(152)
        out.putVarInt64(self.elapsed_usec_)
        if self.has_response_code_:
            out.putVarInt32(160)
            out.putVarInt64(self.response_code_)
        if self.has_retry_reason_:
            out.putVarInt32(218)
            out.putPrefixedString(self.retry_reason_)

    def OutputPartial(self, out):
        if self.has_dispatched_usec_:
            out.putVarInt32(136)
            out.putVarInt64(self.dispatched_usec_)
        if self.has_lag_usec_:
            out.putVarInt32(144)
            out.putVarInt64(self.lag_usec_)
        if self.has_elapsed_usec_:
            out.putVarInt32(152)
            out.putVarInt64(self.elapsed_usec_)
        if self.has_response_code_:
            out.putVarInt32(160)
            out.putVarInt64(self.response_code_)
        if self.has_retry_reason_:
            out.putVarInt32(218)
            out.putPrefixedString(self.retry_reason_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 132:
                break
            if tt == 136:
                self.set_dispatched_usec(d.getVarInt64())
                continue
            if tt == 144:
                self.set_lag_usec(d.getVarInt64())
                continue
            if tt == 152:
                self.set_elapsed_usec(d.getVarInt64())
                continue
            if tt == 160:
                self.set_response_code(d.getVarInt64())
                continue
            if tt == 218:
                self.set_retry_reason(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_dispatched_usec_:
            res += prefix + 'dispatched_usec: %s\n' % self.DebugFormatInt64(self.dispatched_usec_)
        if self.has_lag_usec_:
            res += prefix + 'lag_usec: %s\n' % self.DebugFormatInt64(self.lag_usec_)
        if self.has_elapsed_usec_:
            res += prefix + 'elapsed_usec: %s\n' % self.DebugFormatInt64(self.elapsed_usec_)
        if self.has_response_code_:
            res += prefix + 'response_code: %s\n' % self.DebugFormatInt64(self.response_code_)
        if self.has_retry_reason_:
            res += prefix + 'retry_reason: %s\n' % self.DebugFormatString(self.retry_reason_)
        return res