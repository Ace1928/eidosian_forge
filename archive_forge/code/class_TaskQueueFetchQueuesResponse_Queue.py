from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueFetchQueuesResponse_Queue(ProtocolBuffer.ProtocolMessage):
    has_queue_name_ = 0
    queue_name_ = ''
    has_bucket_refill_per_second_ = 0
    bucket_refill_per_second_ = 0.0
    has_bucket_capacity_ = 0
    bucket_capacity_ = 0.0
    has_user_specified_rate_ = 0
    user_specified_rate_ = ''
    has_paused_ = 0
    paused_ = 0
    has_retry_parameters_ = 0
    retry_parameters_ = None
    has_max_concurrent_requests_ = 0
    max_concurrent_requests_ = 0
    has_mode_ = 0
    mode_ = 0
    has_acl_ = 0
    acl_ = None
    has_creator_name_ = 0
    creator_name_ = 'apphosting'

    def __init__(self, contents=None):
        self.header_override_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def queue_name(self):
        return self.queue_name_

    def set_queue_name(self, x):
        self.has_queue_name_ = 1
        self.queue_name_ = x

    def clear_queue_name(self):
        if self.has_queue_name_:
            self.has_queue_name_ = 0
            self.queue_name_ = ''

    def has_queue_name(self):
        return self.has_queue_name_

    def bucket_refill_per_second(self):
        return self.bucket_refill_per_second_

    def set_bucket_refill_per_second(self, x):
        self.has_bucket_refill_per_second_ = 1
        self.bucket_refill_per_second_ = x

    def clear_bucket_refill_per_second(self):
        if self.has_bucket_refill_per_second_:
            self.has_bucket_refill_per_second_ = 0
            self.bucket_refill_per_second_ = 0.0

    def has_bucket_refill_per_second(self):
        return self.has_bucket_refill_per_second_

    def bucket_capacity(self):
        return self.bucket_capacity_

    def set_bucket_capacity(self, x):
        self.has_bucket_capacity_ = 1
        self.bucket_capacity_ = x

    def clear_bucket_capacity(self):
        if self.has_bucket_capacity_:
            self.has_bucket_capacity_ = 0
            self.bucket_capacity_ = 0.0

    def has_bucket_capacity(self):
        return self.has_bucket_capacity_

    def user_specified_rate(self):
        return self.user_specified_rate_

    def set_user_specified_rate(self, x):
        self.has_user_specified_rate_ = 1
        self.user_specified_rate_ = x

    def clear_user_specified_rate(self):
        if self.has_user_specified_rate_:
            self.has_user_specified_rate_ = 0
            self.user_specified_rate_ = ''

    def has_user_specified_rate(self):
        return self.has_user_specified_rate_

    def paused(self):
        return self.paused_

    def set_paused(self, x):
        self.has_paused_ = 1
        self.paused_ = x

    def clear_paused(self):
        if self.has_paused_:
            self.has_paused_ = 0
            self.paused_ = 0

    def has_paused(self):
        return self.has_paused_

    def retry_parameters(self):
        if self.retry_parameters_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.retry_parameters_ is None:
                    self.retry_parameters_ = TaskQueueRetryParameters()
            finally:
                self.lazy_init_lock_.release()
        return self.retry_parameters_

    def mutable_retry_parameters(self):
        self.has_retry_parameters_ = 1
        return self.retry_parameters()

    def clear_retry_parameters(self):
        if self.has_retry_parameters_:
            self.has_retry_parameters_ = 0
            if self.retry_parameters_ is not None:
                self.retry_parameters_.Clear()

    def has_retry_parameters(self):
        return self.has_retry_parameters_

    def max_concurrent_requests(self):
        return self.max_concurrent_requests_

    def set_max_concurrent_requests(self, x):
        self.has_max_concurrent_requests_ = 1
        self.max_concurrent_requests_ = x

    def clear_max_concurrent_requests(self):
        if self.has_max_concurrent_requests_:
            self.has_max_concurrent_requests_ = 0
            self.max_concurrent_requests_ = 0

    def has_max_concurrent_requests(self):
        return self.has_max_concurrent_requests_

    def mode(self):
        return self.mode_

    def set_mode(self, x):
        self.has_mode_ = 1
        self.mode_ = x

    def clear_mode(self):
        if self.has_mode_:
            self.has_mode_ = 0
            self.mode_ = 0

    def has_mode(self):
        return self.has_mode_

    def acl(self):
        if self.acl_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.acl_ is None:
                    self.acl_ = TaskQueueAcl()
            finally:
                self.lazy_init_lock_.release()
        return self.acl_

    def mutable_acl(self):
        self.has_acl_ = 1
        return self.acl()

    def clear_acl(self):
        if self.has_acl_:
            self.has_acl_ = 0
            if self.acl_ is not None:
                self.acl_.Clear()

    def has_acl(self):
        return self.has_acl_

    def header_override_size(self):
        return len(self.header_override_)

    def header_override_list(self):
        return self.header_override_

    def header_override(self, i):
        return self.header_override_[i]

    def mutable_header_override(self, i):
        return self.header_override_[i]

    def add_header_override(self):
        x = TaskQueueHttpHeader()
        self.header_override_.append(x)
        return x

    def clear_header_override(self):
        self.header_override_ = []

    def creator_name(self):
        return self.creator_name_

    def set_creator_name(self, x):
        self.has_creator_name_ = 1
        self.creator_name_ = x

    def clear_creator_name(self):
        if self.has_creator_name_:
            self.has_creator_name_ = 0
            self.creator_name_ = 'apphosting'

    def has_creator_name(self):
        return self.has_creator_name_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_bucket_refill_per_second():
            self.set_bucket_refill_per_second(x.bucket_refill_per_second())
        if x.has_bucket_capacity():
            self.set_bucket_capacity(x.bucket_capacity())
        if x.has_user_specified_rate():
            self.set_user_specified_rate(x.user_specified_rate())
        if x.has_paused():
            self.set_paused(x.paused())
        if x.has_retry_parameters():
            self.mutable_retry_parameters().MergeFrom(x.retry_parameters())
        if x.has_max_concurrent_requests():
            self.set_max_concurrent_requests(x.max_concurrent_requests())
        if x.has_mode():
            self.set_mode(x.mode())
        if x.has_acl():
            self.mutable_acl().MergeFrom(x.acl())
        for i in range(x.header_override_size()):
            self.add_header_override().CopyFrom(x.header_override(i))
        if x.has_creator_name():
            self.set_creator_name(x.creator_name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_queue_name_ != x.has_queue_name_:
            return 0
        if self.has_queue_name_ and self.queue_name_ != x.queue_name_:
            return 0
        if self.has_bucket_refill_per_second_ != x.has_bucket_refill_per_second_:
            return 0
        if self.has_bucket_refill_per_second_ and self.bucket_refill_per_second_ != x.bucket_refill_per_second_:
            return 0
        if self.has_bucket_capacity_ != x.has_bucket_capacity_:
            return 0
        if self.has_bucket_capacity_ and self.bucket_capacity_ != x.bucket_capacity_:
            return 0
        if self.has_user_specified_rate_ != x.has_user_specified_rate_:
            return 0
        if self.has_user_specified_rate_ and self.user_specified_rate_ != x.user_specified_rate_:
            return 0
        if self.has_paused_ != x.has_paused_:
            return 0
        if self.has_paused_ and self.paused_ != x.paused_:
            return 0
        if self.has_retry_parameters_ != x.has_retry_parameters_:
            return 0
        if self.has_retry_parameters_ and self.retry_parameters_ != x.retry_parameters_:
            return 0
        if self.has_max_concurrent_requests_ != x.has_max_concurrent_requests_:
            return 0
        if self.has_max_concurrent_requests_ and self.max_concurrent_requests_ != x.max_concurrent_requests_:
            return 0
        if self.has_mode_ != x.has_mode_:
            return 0
        if self.has_mode_ and self.mode_ != x.mode_:
            return 0
        if self.has_acl_ != x.has_acl_:
            return 0
        if self.has_acl_ and self.acl_ != x.acl_:
            return 0
        if len(self.header_override_) != len(x.header_override_):
            return 0
        for e1, e2 in zip(self.header_override_, x.header_override_):
            if e1 != e2:
                return 0
        if self.has_creator_name_ != x.has_creator_name_:
            return 0
        if self.has_creator_name_ and self.creator_name_ != x.creator_name_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_queue_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: queue_name not set.')
        if not self.has_bucket_refill_per_second_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: bucket_refill_per_second not set.')
        if not self.has_bucket_capacity_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: bucket_capacity not set.')
        if not self.has_paused_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: paused not set.')
        if self.has_retry_parameters_ and (not self.retry_parameters_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_acl_ and (not self.acl_.IsInitialized(debug_strs)):
            initialized = 0
        for p in self.header_override_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.queue_name_))
        if self.has_user_specified_rate_:
            n += 1 + self.lengthString(len(self.user_specified_rate_))
        if self.has_retry_parameters_:
            n += 1 + self.lengthString(self.retry_parameters_.ByteSize())
        if self.has_max_concurrent_requests_:
            n += 1 + self.lengthVarInt64(self.max_concurrent_requests_)
        if self.has_mode_:
            n += 1 + self.lengthVarInt64(self.mode_)
        if self.has_acl_:
            n += 1 + self.lengthString(self.acl_.ByteSize())
        n += 1 * len(self.header_override_)
        for i in range(len(self.header_override_)):
            n += self.lengthString(self.header_override_[i].ByteSize())
        if self.has_creator_name_:
            n += 1 + self.lengthString(len(self.creator_name_))
        return n + 21

    def ByteSizePartial(self):
        n = 0
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_bucket_refill_per_second_:
            n += 9
        if self.has_bucket_capacity_:
            n += 9
        if self.has_user_specified_rate_:
            n += 1 + self.lengthString(len(self.user_specified_rate_))
        if self.has_paused_:
            n += 2
        if self.has_retry_parameters_:
            n += 1 + self.lengthString(self.retry_parameters_.ByteSizePartial())
        if self.has_max_concurrent_requests_:
            n += 1 + self.lengthVarInt64(self.max_concurrent_requests_)
        if self.has_mode_:
            n += 1 + self.lengthVarInt64(self.mode_)
        if self.has_acl_:
            n += 1 + self.lengthString(self.acl_.ByteSizePartial())
        n += 1 * len(self.header_override_)
        for i in range(len(self.header_override_)):
            n += self.lengthString(self.header_override_[i].ByteSizePartial())
        if self.has_creator_name_:
            n += 1 + self.lengthString(len(self.creator_name_))
        return n

    def Clear(self):
        self.clear_queue_name()
        self.clear_bucket_refill_per_second()
        self.clear_bucket_capacity()
        self.clear_user_specified_rate()
        self.clear_paused()
        self.clear_retry_parameters()
        self.clear_max_concurrent_requests()
        self.clear_mode()
        self.clear_acl()
        self.clear_header_override()
        self.clear_creator_name()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(25)
        out.putDouble(self.bucket_refill_per_second_)
        out.putVarInt32(33)
        out.putDouble(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            out.putVarInt32(42)
            out.putPrefixedString(self.user_specified_rate_)
        out.putVarInt32(48)
        out.putBoolean(self.paused_)
        if self.has_retry_parameters_:
            out.putVarInt32(58)
            out.putVarInt32(self.retry_parameters_.ByteSize())
            self.retry_parameters_.OutputUnchecked(out)
        if self.has_max_concurrent_requests_:
            out.putVarInt32(64)
            out.putVarInt32(self.max_concurrent_requests_)
        if self.has_mode_:
            out.putVarInt32(72)
            out.putVarInt32(self.mode_)
        if self.has_acl_:
            out.putVarInt32(82)
            out.putVarInt32(self.acl_.ByteSize())
            self.acl_.OutputUnchecked(out)
        for i in range(len(self.header_override_)):
            out.putVarInt32(90)
            out.putVarInt32(self.header_override_[i].ByteSize())
            self.header_override_[i].OutputUnchecked(out)
        if self.has_creator_name_:
            out.putVarInt32(98)
            out.putPrefixedString(self.creator_name_)

    def OutputPartial(self, out):
        if self.has_queue_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_)
        if self.has_bucket_refill_per_second_:
            out.putVarInt32(25)
            out.putDouble(self.bucket_refill_per_second_)
        if self.has_bucket_capacity_:
            out.putVarInt32(33)
            out.putDouble(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            out.putVarInt32(42)
            out.putPrefixedString(self.user_specified_rate_)
        if self.has_paused_:
            out.putVarInt32(48)
            out.putBoolean(self.paused_)
        if self.has_retry_parameters_:
            out.putVarInt32(58)
            out.putVarInt32(self.retry_parameters_.ByteSizePartial())
            self.retry_parameters_.OutputPartial(out)
        if self.has_max_concurrent_requests_:
            out.putVarInt32(64)
            out.putVarInt32(self.max_concurrent_requests_)
        if self.has_mode_:
            out.putVarInt32(72)
            out.putVarInt32(self.mode_)
        if self.has_acl_:
            out.putVarInt32(82)
            out.putVarInt32(self.acl_.ByteSizePartial())
            self.acl_.OutputPartial(out)
        for i in range(len(self.header_override_)):
            out.putVarInt32(90)
            out.putVarInt32(self.header_override_[i].ByteSizePartial())
            self.header_override_[i].OutputPartial(out)
        if self.has_creator_name_:
            out.putVarInt32(98)
            out.putPrefixedString(self.creator_name_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 25:
                self.set_bucket_refill_per_second(d.getDouble())
                continue
            if tt == 33:
                self.set_bucket_capacity(d.getDouble())
                continue
            if tt == 42:
                self.set_user_specified_rate(d.getPrefixedString())
                continue
            if tt == 48:
                self.set_paused(d.getBoolean())
                continue
            if tt == 58:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_retry_parameters().TryMerge(tmp)
                continue
            if tt == 64:
                self.set_max_concurrent_requests(d.getVarInt32())
                continue
            if tt == 72:
                self.set_mode(d.getVarInt32())
                continue
            if tt == 82:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_acl().TryMerge(tmp)
                continue
            if tt == 90:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_header_override().TryMerge(tmp)
                continue
            if tt == 98:
                self.set_creator_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_bucket_refill_per_second_:
            res += prefix + 'bucket_refill_per_second: %s\n' % self.DebugFormat(self.bucket_refill_per_second_)
        if self.has_bucket_capacity_:
            res += prefix + 'bucket_capacity: %s\n' % self.DebugFormat(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            res += prefix + 'user_specified_rate: %s\n' % self.DebugFormatString(self.user_specified_rate_)
        if self.has_paused_:
            res += prefix + 'paused: %s\n' % self.DebugFormatBool(self.paused_)
        if self.has_retry_parameters_:
            res += prefix + 'retry_parameters <\n'
            res += self.retry_parameters_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_max_concurrent_requests_:
            res += prefix + 'max_concurrent_requests: %s\n' % self.DebugFormatInt32(self.max_concurrent_requests_)
        if self.has_mode_:
            res += prefix + 'mode: %s\n' % self.DebugFormatInt32(self.mode_)
        if self.has_acl_:
            res += prefix + 'acl <\n'
            res += self.acl_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        cnt = 0
        for e in self.header_override_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'header_override%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_creator_name_:
            res += prefix + 'creator_name: %s\n' % self.DebugFormatString(self.creator_name_)
        return res