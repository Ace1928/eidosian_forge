from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueAddRequest(ProtocolBuffer.ProtocolMessage):
    GET = 1
    POST = 2
    HEAD = 3
    PUT = 4
    DELETE = 5
    _RequestMethod_NAMES = {1: 'GET', 2: 'POST', 3: 'HEAD', 4: 'PUT', 5: 'DELETE'}

    def RequestMethod_Name(cls, x):
        return cls._RequestMethod_NAMES.get(x, '')
    RequestMethod_Name = classmethod(RequestMethod_Name)
    has_queue_name_ = 0
    queue_name_ = ''
    has_task_name_ = 0
    task_name_ = ''
    has_eta_usec_ = 0
    eta_usec_ = 0
    has_method_ = 0
    method_ = 2
    has_url_ = 0
    url_ = ''
    has_body_ = 0
    body_ = ''
    has_transaction_ = 0
    transaction_ = None
    has_datastore_transaction_ = 0
    datastore_transaction_ = ''
    has_app_id_ = 0
    app_id_ = ''
    has_crontimetable_ = 0
    crontimetable_ = None
    has_description_ = 0
    description_ = ''
    has_payload_ = 0
    payload_ = None
    has_retry_parameters_ = 0
    retry_parameters_ = None
    has_mode_ = 0
    mode_ = 0
    has_tag_ = 0
    tag_ = ''
    has_cron_retry_parameters_ = 0
    cron_retry_parameters_ = None

    def __init__(self, contents=None):
        self.header_ = []
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

    def task_name(self):
        return self.task_name_

    def set_task_name(self, x):
        self.has_task_name_ = 1
        self.task_name_ = x

    def clear_task_name(self):
        if self.has_task_name_:
            self.has_task_name_ = 0
            self.task_name_ = ''

    def has_task_name(self):
        return self.has_task_name_

    def eta_usec(self):
        return self.eta_usec_

    def set_eta_usec(self, x):
        self.has_eta_usec_ = 1
        self.eta_usec_ = x

    def clear_eta_usec(self):
        if self.has_eta_usec_:
            self.has_eta_usec_ = 0
            self.eta_usec_ = 0

    def has_eta_usec(self):
        return self.has_eta_usec_

    def method(self):
        return self.method_

    def set_method(self, x):
        self.has_method_ = 1
        self.method_ = x

    def clear_method(self):
        if self.has_method_:
            self.has_method_ = 0
            self.method_ = 2

    def has_method(self):
        return self.has_method_

    def url(self):
        return self.url_

    def set_url(self, x):
        self.has_url_ = 1
        self.url_ = x

    def clear_url(self):
        if self.has_url_:
            self.has_url_ = 0
            self.url_ = ''

    def has_url(self):
        return self.has_url_

    def header_size(self):
        return len(self.header_)

    def header_list(self):
        return self.header_

    def header(self, i):
        return self.header_[i]

    def mutable_header(self, i):
        return self.header_[i]

    def add_header(self):
        x = TaskQueueAddRequest_Header()
        self.header_.append(x)
        return x

    def clear_header(self):
        self.header_ = []

    def body(self):
        return self.body_

    def set_body(self, x):
        self.has_body_ = 1
        self.body_ = x

    def clear_body(self):
        if self.has_body_:
            self.has_body_ = 0
            self.body_ = ''

    def has_body(self):
        return self.has_body_

    def transaction(self):
        if self.transaction_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.transaction_ is None:
                    self.transaction_ = googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb.Transaction()
            finally:
                self.lazy_init_lock_.release()
        return self.transaction_

    def mutable_transaction(self):
        self.has_transaction_ = 1
        return self.transaction()

    def clear_transaction(self):
        if self.has_transaction_:
            self.has_transaction_ = 0
            if self.transaction_ is not None:
                self.transaction_.Clear()

    def has_transaction(self):
        return self.has_transaction_

    def datastore_transaction(self):
        return self.datastore_transaction_

    def set_datastore_transaction(self, x):
        self.has_datastore_transaction_ = 1
        self.datastore_transaction_ = x

    def clear_datastore_transaction(self):
        if self.has_datastore_transaction_:
            self.has_datastore_transaction_ = 0
            self.datastore_transaction_ = ''

    def has_datastore_transaction(self):
        return self.has_datastore_transaction_

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

    def crontimetable(self):
        if self.crontimetable_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.crontimetable_ is None:
                    self.crontimetable_ = TaskQueueAddRequest_CronTimetable()
            finally:
                self.lazy_init_lock_.release()
        return self.crontimetable_

    def mutable_crontimetable(self):
        self.has_crontimetable_ = 1
        return self.crontimetable()

    def clear_crontimetable(self):
        if self.has_crontimetable_:
            self.has_crontimetable_ = 0
            if self.crontimetable_ is not None:
                self.crontimetable_.Clear()

    def has_crontimetable(self):
        return self.has_crontimetable_

    def description(self):
        return self.description_

    def set_description(self, x):
        self.has_description_ = 1
        self.description_ = x

    def clear_description(self):
        if self.has_description_:
            self.has_description_ = 0
            self.description_ = ''

    def has_description(self):
        return self.has_description_

    def payload(self):
        if self.payload_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.payload_ is None:
                    self.payload_ = MessageSet()
            finally:
                self.lazy_init_lock_.release()
        return self.payload_

    def mutable_payload(self):
        self.has_payload_ = 1
        return self.payload()

    def clear_payload(self):
        if self.has_payload_:
            self.has_payload_ = 0
            if self.payload_ is not None:
                self.payload_.Clear()

    def has_payload(self):
        return self.has_payload_

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

    def tag(self):
        return self.tag_

    def set_tag(self, x):
        self.has_tag_ = 1
        self.tag_ = x

    def clear_tag(self):
        if self.has_tag_:
            self.has_tag_ = 0
            self.tag_ = ''

    def has_tag(self):
        return self.has_tag_

    def cron_retry_parameters(self):
        if self.cron_retry_parameters_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.cron_retry_parameters_ is None:
                    self.cron_retry_parameters_ = TaskQueueRetryParameters()
            finally:
                self.lazy_init_lock_.release()
        return self.cron_retry_parameters_

    def mutable_cron_retry_parameters(self):
        self.has_cron_retry_parameters_ = 1
        return self.cron_retry_parameters()

    def clear_cron_retry_parameters(self):
        if self.has_cron_retry_parameters_:
            self.has_cron_retry_parameters_ = 0
            if self.cron_retry_parameters_ is not None:
                self.cron_retry_parameters_.Clear()

    def has_cron_retry_parameters(self):
        return self.has_cron_retry_parameters_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_task_name():
            self.set_task_name(x.task_name())
        if x.has_eta_usec():
            self.set_eta_usec(x.eta_usec())
        if x.has_method():
            self.set_method(x.method())
        if x.has_url():
            self.set_url(x.url())
        for i in range(x.header_size()):
            self.add_header().CopyFrom(x.header(i))
        if x.has_body():
            self.set_body(x.body())
        if x.has_transaction():
            self.mutable_transaction().MergeFrom(x.transaction())
        if x.has_datastore_transaction():
            self.set_datastore_transaction(x.datastore_transaction())
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_crontimetable():
            self.mutable_crontimetable().MergeFrom(x.crontimetable())
        if x.has_description():
            self.set_description(x.description())
        if x.has_payload():
            self.mutable_payload().MergeFrom(x.payload())
        if x.has_retry_parameters():
            self.mutable_retry_parameters().MergeFrom(x.retry_parameters())
        if x.has_mode():
            self.set_mode(x.mode())
        if x.has_tag():
            self.set_tag(x.tag())
        if x.has_cron_retry_parameters():
            self.mutable_cron_retry_parameters().MergeFrom(x.cron_retry_parameters())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_queue_name_ != x.has_queue_name_:
            return 0
        if self.has_queue_name_ and self.queue_name_ != x.queue_name_:
            return 0
        if self.has_task_name_ != x.has_task_name_:
            return 0
        if self.has_task_name_ and self.task_name_ != x.task_name_:
            return 0
        if self.has_eta_usec_ != x.has_eta_usec_:
            return 0
        if self.has_eta_usec_ and self.eta_usec_ != x.eta_usec_:
            return 0
        if self.has_method_ != x.has_method_:
            return 0
        if self.has_method_ and self.method_ != x.method_:
            return 0
        if self.has_url_ != x.has_url_:
            return 0
        if self.has_url_ and self.url_ != x.url_:
            return 0
        if len(self.header_) != len(x.header_):
            return 0
        for e1, e2 in zip(self.header_, x.header_):
            if e1 != e2:
                return 0
        if self.has_body_ != x.has_body_:
            return 0
        if self.has_body_ and self.body_ != x.body_:
            return 0
        if self.has_transaction_ != x.has_transaction_:
            return 0
        if self.has_transaction_ and self.transaction_ != x.transaction_:
            return 0
        if self.has_datastore_transaction_ != x.has_datastore_transaction_:
            return 0
        if self.has_datastore_transaction_ and self.datastore_transaction_ != x.datastore_transaction_:
            return 0
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
        if self.has_crontimetable_ != x.has_crontimetable_:
            return 0
        if self.has_crontimetable_ and self.crontimetable_ != x.crontimetable_:
            return 0
        if self.has_description_ != x.has_description_:
            return 0
        if self.has_description_ and self.description_ != x.description_:
            return 0
        if self.has_payload_ != x.has_payload_:
            return 0
        if self.has_payload_ and self.payload_ != x.payload_:
            return 0
        if self.has_retry_parameters_ != x.has_retry_parameters_:
            return 0
        if self.has_retry_parameters_ and self.retry_parameters_ != x.retry_parameters_:
            return 0
        if self.has_mode_ != x.has_mode_:
            return 0
        if self.has_mode_ and self.mode_ != x.mode_:
            return 0
        if self.has_tag_ != x.has_tag_:
            return 0
        if self.has_tag_ and self.tag_ != x.tag_:
            return 0
        if self.has_cron_retry_parameters_ != x.has_cron_retry_parameters_:
            return 0
        if self.has_cron_retry_parameters_ and self.cron_retry_parameters_ != x.cron_retry_parameters_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_queue_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: queue_name not set.')
        if not self.has_task_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: task_name not set.')
        if not self.has_eta_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: eta_usec not set.')
        for p in self.header_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_transaction_ and (not self.transaction_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_crontimetable_ and (not self.crontimetable_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_payload_ and (not self.payload_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_retry_parameters_ and (not self.retry_parameters_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_cron_retry_parameters_ and (not self.cron_retry_parameters_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.queue_name_))
        n += self.lengthString(len(self.task_name_))
        n += self.lengthVarInt64(self.eta_usec_)
        if self.has_method_:
            n += 1 + self.lengthVarInt64(self.method_)
        if self.has_url_:
            n += 1 + self.lengthString(len(self.url_))
        n += 2 * len(self.header_)
        for i in range(len(self.header_)):
            n += self.header_[i].ByteSize()
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        if self.has_transaction_:
            n += 1 + self.lengthString(self.transaction_.ByteSize())
        if self.has_datastore_transaction_:
            n += 2 + self.lengthString(len(self.datastore_transaction_))
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_crontimetable_:
            n += 2 + self.crontimetable_.ByteSize()
        if self.has_description_:
            n += 1 + self.lengthString(len(self.description_))
        if self.has_payload_:
            n += 2 + self.lengthString(self.payload_.ByteSize())
        if self.has_retry_parameters_:
            n += 2 + self.lengthString(self.retry_parameters_.ByteSize())
        if self.has_mode_:
            n += 2 + self.lengthVarInt64(self.mode_)
        if self.has_tag_:
            n += 2 + self.lengthString(len(self.tag_))
        if self.has_cron_retry_parameters_:
            n += 2 + self.lengthString(self.cron_retry_parameters_.ByteSize())
        return n + 3

    def ByteSizePartial(self):
        n = 0
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_task_name_:
            n += 1
            n += self.lengthString(len(self.task_name_))
        if self.has_eta_usec_:
            n += 1
            n += self.lengthVarInt64(self.eta_usec_)
        if self.has_method_:
            n += 1 + self.lengthVarInt64(self.method_)
        if self.has_url_:
            n += 1 + self.lengthString(len(self.url_))
        n += 2 * len(self.header_)
        for i in range(len(self.header_)):
            n += self.header_[i].ByteSizePartial()
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        if self.has_transaction_:
            n += 1 + self.lengthString(self.transaction_.ByteSizePartial())
        if self.has_datastore_transaction_:
            n += 2 + self.lengthString(len(self.datastore_transaction_))
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_crontimetable_:
            n += 2 + self.crontimetable_.ByteSizePartial()
        if self.has_description_:
            n += 1 + self.lengthString(len(self.description_))
        if self.has_payload_:
            n += 2 + self.lengthString(self.payload_.ByteSizePartial())
        if self.has_retry_parameters_:
            n += 2 + self.lengthString(self.retry_parameters_.ByteSizePartial())
        if self.has_mode_:
            n += 2 + self.lengthVarInt64(self.mode_)
        if self.has_tag_:
            n += 2 + self.lengthString(len(self.tag_))
        if self.has_cron_retry_parameters_:
            n += 2 + self.lengthString(self.cron_retry_parameters_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_queue_name()
        self.clear_task_name()
        self.clear_eta_usec()
        self.clear_method()
        self.clear_url()
        self.clear_header()
        self.clear_body()
        self.clear_transaction()
        self.clear_datastore_transaction()
        self.clear_app_id()
        self.clear_crontimetable()
        self.clear_description()
        self.clear_payload()
        self.clear_retry_parameters()
        self.clear_mode()
        self.clear_tag()
        self.clear_cron_retry_parameters()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(18)
        out.putPrefixedString(self.task_name_)
        out.putVarInt32(24)
        out.putVarInt64(self.eta_usec_)
        if self.has_url_:
            out.putVarInt32(34)
            out.putPrefixedString(self.url_)
        if self.has_method_:
            out.putVarInt32(40)
            out.putVarInt32(self.method_)
        for i in range(len(self.header_)):
            out.putVarInt32(51)
            self.header_[i].OutputUnchecked(out)
            out.putVarInt32(52)
        if self.has_body_:
            out.putVarInt32(74)
            out.putPrefixedString(self.body_)
        if self.has_transaction_:
            out.putVarInt32(82)
            out.putVarInt32(self.transaction_.ByteSize())
            self.transaction_.OutputUnchecked(out)
        if self.has_app_id_:
            out.putVarInt32(90)
            out.putPrefixedString(self.app_id_)
        if self.has_crontimetable_:
            out.putVarInt32(99)
            self.crontimetable_.OutputUnchecked(out)
            out.putVarInt32(100)
        if self.has_description_:
            out.putVarInt32(122)
            out.putPrefixedString(self.description_)
        if self.has_payload_:
            out.putVarInt32(130)
            out.putVarInt32(self.payload_.ByteSize())
            self.payload_.OutputUnchecked(out)
        if self.has_retry_parameters_:
            out.putVarInt32(138)
            out.putVarInt32(self.retry_parameters_.ByteSize())
            self.retry_parameters_.OutputUnchecked(out)
        if self.has_mode_:
            out.putVarInt32(144)
            out.putVarInt32(self.mode_)
        if self.has_tag_:
            out.putVarInt32(154)
            out.putPrefixedString(self.tag_)
        if self.has_cron_retry_parameters_:
            out.putVarInt32(162)
            out.putVarInt32(self.cron_retry_parameters_.ByteSize())
            self.cron_retry_parameters_.OutputUnchecked(out)
        if self.has_datastore_transaction_:
            out.putVarInt32(170)
            out.putPrefixedString(self.datastore_transaction_)

    def OutputPartial(self, out):
        if self.has_queue_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.queue_name_)
        if self.has_task_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.task_name_)
        if self.has_eta_usec_:
            out.putVarInt32(24)
            out.putVarInt64(self.eta_usec_)
        if self.has_url_:
            out.putVarInt32(34)
            out.putPrefixedString(self.url_)
        if self.has_method_:
            out.putVarInt32(40)
            out.putVarInt32(self.method_)
        for i in range(len(self.header_)):
            out.putVarInt32(51)
            self.header_[i].OutputPartial(out)
            out.putVarInt32(52)
        if self.has_body_:
            out.putVarInt32(74)
            out.putPrefixedString(self.body_)
        if self.has_transaction_:
            out.putVarInt32(82)
            out.putVarInt32(self.transaction_.ByteSizePartial())
            self.transaction_.OutputPartial(out)
        if self.has_app_id_:
            out.putVarInt32(90)
            out.putPrefixedString(self.app_id_)
        if self.has_crontimetable_:
            out.putVarInt32(99)
            self.crontimetable_.OutputPartial(out)
            out.putVarInt32(100)
        if self.has_description_:
            out.putVarInt32(122)
            out.putPrefixedString(self.description_)
        if self.has_payload_:
            out.putVarInt32(130)
            out.putVarInt32(self.payload_.ByteSizePartial())
            self.payload_.OutputPartial(out)
        if self.has_retry_parameters_:
            out.putVarInt32(138)
            out.putVarInt32(self.retry_parameters_.ByteSizePartial())
            self.retry_parameters_.OutputPartial(out)
        if self.has_mode_:
            out.putVarInt32(144)
            out.putVarInt32(self.mode_)
        if self.has_tag_:
            out.putVarInt32(154)
            out.putPrefixedString(self.tag_)
        if self.has_cron_retry_parameters_:
            out.putVarInt32(162)
            out.putVarInt32(self.cron_retry_parameters_.ByteSizePartial())
            self.cron_retry_parameters_.OutputPartial(out)
        if self.has_datastore_transaction_:
            out.putVarInt32(170)
            out.putPrefixedString(self.datastore_transaction_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_task_name(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_eta_usec(d.getVarInt64())
                continue
            if tt == 34:
                self.set_url(d.getPrefixedString())
                continue
            if tt == 40:
                self.set_method(d.getVarInt32())
                continue
            if tt == 51:
                self.add_header().TryMerge(d)
                continue
            if tt == 74:
                self.set_body(d.getPrefixedString())
                continue
            if tt == 82:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_transaction().TryMerge(tmp)
                continue
            if tt == 90:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 99:
                self.mutable_crontimetable().TryMerge(d)
                continue
            if tt == 122:
                self.set_description(d.getPrefixedString())
                continue
            if tt == 130:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_payload().TryMerge(tmp)
                continue
            if tt == 138:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_retry_parameters().TryMerge(tmp)
                continue
            if tt == 144:
                self.set_mode(d.getVarInt32())
                continue
            if tt == 154:
                self.set_tag(d.getPrefixedString())
                continue
            if tt == 162:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_cron_retry_parameters().TryMerge(tmp)
                continue
            if tt == 170:
                self.set_datastore_transaction(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_task_name_:
            res += prefix + 'task_name: %s\n' % self.DebugFormatString(self.task_name_)
        if self.has_eta_usec_:
            res += prefix + 'eta_usec: %s\n' % self.DebugFormatInt64(self.eta_usec_)
        if self.has_method_:
            res += prefix + 'method: %s\n' % self.DebugFormatInt32(self.method_)
        if self.has_url_:
            res += prefix + 'url: %s\n' % self.DebugFormatString(self.url_)
        cnt = 0
        for e in self.header_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Header%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        if self.has_body_:
            res += prefix + 'body: %s\n' % self.DebugFormatString(self.body_)
        if self.has_transaction_:
            res += prefix + 'transaction <\n'
            res += self.transaction_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_datastore_transaction_:
            res += prefix + 'datastore_transaction: %s\n' % self.DebugFormatString(self.datastore_transaction_)
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        if self.has_crontimetable_:
            res += prefix + 'CronTimetable {\n'
            res += self.crontimetable_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_description_:
            res += prefix + 'description: %s\n' % self.DebugFormatString(self.description_)
        if self.has_payload_:
            res += prefix + 'payload <\n'
            res += self.payload_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_retry_parameters_:
            res += prefix + 'retry_parameters <\n'
            res += self.retry_parameters_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_mode_:
            res += prefix + 'mode: %s\n' % self.DebugFormatInt32(self.mode_)
        if self.has_tag_:
            res += prefix + 'tag: %s\n' % self.DebugFormatString(self.tag_)
        if self.has_cron_retry_parameters_:
            res += prefix + 'cron_retry_parameters <\n'
            res += self.cron_retry_parameters_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kqueue_name = 1
    ktask_name = 2
    keta_usec = 3
    kmethod = 5
    kurl = 4
    kHeaderGroup = 6
    kHeaderkey = 7
    kHeadervalue = 8
    kbody = 9
    ktransaction = 10
    kdatastore_transaction = 21
    kapp_id = 11
    kCronTimetableGroup = 12
    kCronTimetableschedule = 13
    kCronTimetabletimezone = 14
    kdescription = 15
    kpayload = 16
    kretry_parameters = 17
    kmode = 18
    ktag = 19
    kcron_retry_parameters = 20
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'queue_name', 2: 'task_name', 3: 'eta_usec', 4: 'url', 5: 'method', 6: 'Header', 7: 'key', 8: 'value', 9: 'body', 10: 'transaction', 11: 'app_id', 12: 'CronTimetable', 13: 'schedule', 14: 'timezone', 15: 'description', 16: 'payload', 17: 'retry_parameters', 18: 'mode', 19: 'tag', 20: 'cron_retry_parameters', 21: 'datastore_transaction'}, 21)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.STARTGROUP, 7: ProtocolBuffer.Encoder.STRING, 8: ProtocolBuffer.Encoder.STRING, 9: ProtocolBuffer.Encoder.STRING, 10: ProtocolBuffer.Encoder.STRING, 11: ProtocolBuffer.Encoder.STRING, 12: ProtocolBuffer.Encoder.STARTGROUP, 13: ProtocolBuffer.Encoder.STRING, 14: ProtocolBuffer.Encoder.STRING, 15: ProtocolBuffer.Encoder.STRING, 16: ProtocolBuffer.Encoder.STRING, 17: ProtocolBuffer.Encoder.STRING, 18: ProtocolBuffer.Encoder.NUMERIC, 19: ProtocolBuffer.Encoder.STRING, 20: ProtocolBuffer.Encoder.STRING, 21: ProtocolBuffer.Encoder.STRING}, 21, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueAddRequest'