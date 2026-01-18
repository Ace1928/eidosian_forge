from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueUpdateQueueRequest(ProtocolBuffer.ProtocolMessage):
    has_app_id_ = 0
    app_id_ = ''
    has_queue_name_ = 0
    queue_name_ = ''
    has_bucket_refill_per_second_ = 0
    bucket_refill_per_second_ = 0.0
    has_bucket_capacity_ = 0
    bucket_capacity_ = 0
    has_user_specified_rate_ = 0
    user_specified_rate_ = ''
    has_retry_parameters_ = 0
    retry_parameters_ = None
    has_max_concurrent_requests_ = 0
    max_concurrent_requests_ = 0
    has_mode_ = 0
    mode_ = 0
    has_acl_ = 0
    acl_ = None

    def __init__(self, contents=None):
        self.header_override_ = []
        self.lazy_init_lock_ = _Lock()
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
            self.bucket_capacity_ = 0

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

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_bucket_refill_per_second():
            self.set_bucket_refill_per_second(x.bucket_refill_per_second())
        if x.has_bucket_capacity():
            self.set_bucket_capacity(x.bucket_capacity())
        if x.has_user_specified_rate():
            self.set_user_specified_rate(x.user_specified_rate())
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

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
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
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += self.lengthString(len(self.queue_name_))
        n += self.lengthVarInt64(self.bucket_capacity_)
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
        return n + 11

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_bucket_refill_per_second_:
            n += 9
        if self.has_bucket_capacity_:
            n += 1
            n += self.lengthVarInt64(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            n += 1 + self.lengthString(len(self.user_specified_rate_))
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
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_queue_name()
        self.clear_bucket_refill_per_second()
        self.clear_bucket_capacity()
        self.clear_user_specified_rate()
        self.clear_retry_parameters()
        self.clear_max_concurrent_requests()
        self.clear_mode()
        self.clear_acl()
        self.clear_header_override()

    def OutputUnchecked(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        out.putVarInt32(18)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(25)
        out.putDouble(self.bucket_refill_per_second_)
        out.putVarInt32(32)
        out.putVarInt32(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            out.putVarInt32(42)
            out.putPrefixedString(self.user_specified_rate_)
        if self.has_retry_parameters_:
            out.putVarInt32(50)
            out.putVarInt32(self.retry_parameters_.ByteSize())
            self.retry_parameters_.OutputUnchecked(out)
        if self.has_max_concurrent_requests_:
            out.putVarInt32(56)
            out.putVarInt32(self.max_concurrent_requests_)
        if self.has_mode_:
            out.putVarInt32(64)
            out.putVarInt32(self.mode_)
        if self.has_acl_:
            out.putVarInt32(74)
            out.putVarInt32(self.acl_.ByteSize())
            self.acl_.OutputUnchecked(out)
        for i in range(len(self.header_override_)):
            out.putVarInt32(82)
            out.putVarInt32(self.header_override_[i].ByteSize())
            self.header_override_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        if self.has_queue_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_)
        if self.has_bucket_refill_per_second_:
            out.putVarInt32(25)
            out.putDouble(self.bucket_refill_per_second_)
        if self.has_bucket_capacity_:
            out.putVarInt32(32)
            out.putVarInt32(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            out.putVarInt32(42)
            out.putPrefixedString(self.user_specified_rate_)
        if self.has_retry_parameters_:
            out.putVarInt32(50)
            out.putVarInt32(self.retry_parameters_.ByteSizePartial())
            self.retry_parameters_.OutputPartial(out)
        if self.has_max_concurrent_requests_:
            out.putVarInt32(56)
            out.putVarInt32(self.max_concurrent_requests_)
        if self.has_mode_:
            out.putVarInt32(64)
            out.putVarInt32(self.mode_)
        if self.has_acl_:
            out.putVarInt32(74)
            out.putVarInt32(self.acl_.ByteSizePartial())
            self.acl_.OutputPartial(out)
        for i in range(len(self.header_override_)):
            out.putVarInt32(82)
            out.putVarInt32(self.header_override_[i].ByteSizePartial())
            self.header_override_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 25:
                self.set_bucket_refill_per_second(d.getDouble())
                continue
            if tt == 32:
                self.set_bucket_capacity(d.getVarInt32())
                continue
            if tt == 42:
                self.set_user_specified_rate(d.getPrefixedString())
                continue
            if tt == 50:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_retry_parameters().TryMerge(tmp)
                continue
            if tt == 56:
                self.set_max_concurrent_requests(d.getVarInt32())
                continue
            if tt == 64:
                self.set_mode(d.getVarInt32())
                continue
            if tt == 74:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_acl().TryMerge(tmp)
                continue
            if tt == 82:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_header_override().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_bucket_refill_per_second_:
            res += prefix + 'bucket_refill_per_second: %s\n' % self.DebugFormat(self.bucket_refill_per_second_)
        if self.has_bucket_capacity_:
            res += prefix + 'bucket_capacity: %s\n' % self.DebugFormatInt32(self.bucket_capacity_)
        if self.has_user_specified_rate_:
            res += prefix + 'user_specified_rate: %s\n' % self.DebugFormatString(self.user_specified_rate_)
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
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kqueue_name = 2
    kbucket_refill_per_second = 3
    kbucket_capacity = 4
    kuser_specified_rate = 5
    kretry_parameters = 6
    kmax_concurrent_requests = 7
    kmode = 8
    kacl = 9
    kheader_override = 10
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'queue_name', 3: 'bucket_refill_per_second', 4: 'bucket_capacity', 5: 'user_specified_rate', 6: 'retry_parameters', 7: 'max_concurrent_requests', 8: 'mode', 9: 'acl', 10: 'header_override'}, 10)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.DOUBLE, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STRING, 6: ProtocolBuffer.Encoder.STRING, 7: ProtocolBuffer.Encoder.NUMERIC, 8: ProtocolBuffer.Encoder.NUMERIC, 9: ProtocolBuffer.Encoder.STRING, 10: ProtocolBuffer.Encoder.STRING}, 10, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueUpdateQueueRequest'