from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryTasksRequest(ProtocolBuffer.ProtocolMessage):
    has_app_id_ = 0
    app_id_ = ''
    has_queue_name_ = 0
    queue_name_ = ''
    has_start_task_name_ = 0
    start_task_name_ = ''
    has_start_eta_usec_ = 0
    start_eta_usec_ = 0
    has_start_tag_ = 0
    start_tag_ = ''
    has_max_rows_ = 0
    max_rows_ = 1

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

    def start_task_name(self):
        return self.start_task_name_

    def set_start_task_name(self, x):
        self.has_start_task_name_ = 1
        self.start_task_name_ = x

    def clear_start_task_name(self):
        if self.has_start_task_name_:
            self.has_start_task_name_ = 0
            self.start_task_name_ = ''

    def has_start_task_name(self):
        return self.has_start_task_name_

    def start_eta_usec(self):
        return self.start_eta_usec_

    def set_start_eta_usec(self, x):
        self.has_start_eta_usec_ = 1
        self.start_eta_usec_ = x

    def clear_start_eta_usec(self):
        if self.has_start_eta_usec_:
            self.has_start_eta_usec_ = 0
            self.start_eta_usec_ = 0

    def has_start_eta_usec(self):
        return self.has_start_eta_usec_

    def start_tag(self):
        return self.start_tag_

    def set_start_tag(self, x):
        self.has_start_tag_ = 1
        self.start_tag_ = x

    def clear_start_tag(self):
        if self.has_start_tag_:
            self.has_start_tag_ = 0
            self.start_tag_ = ''

    def has_start_tag(self):
        return self.has_start_tag_

    def max_rows(self):
        return self.max_rows_

    def set_max_rows(self, x):
        self.has_max_rows_ = 1
        self.max_rows_ = x

    def clear_max_rows(self):
        if self.has_max_rows_:
            self.has_max_rows_ = 0
            self.max_rows_ = 1

    def has_max_rows(self):
        return self.has_max_rows_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_start_task_name():
            self.set_start_task_name(x.start_task_name())
        if x.has_start_eta_usec():
            self.set_start_eta_usec(x.start_eta_usec())
        if x.has_start_tag():
            self.set_start_tag(x.start_tag())
        if x.has_max_rows():
            self.set_max_rows(x.max_rows())

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
        if self.has_start_task_name_ != x.has_start_task_name_:
            return 0
        if self.has_start_task_name_ and self.start_task_name_ != x.start_task_name_:
            return 0
        if self.has_start_eta_usec_ != x.has_start_eta_usec_:
            return 0
        if self.has_start_eta_usec_ and self.start_eta_usec_ != x.start_eta_usec_:
            return 0
        if self.has_start_tag_ != x.has_start_tag_:
            return 0
        if self.has_start_tag_ and self.start_tag_ != x.start_tag_:
            return 0
        if self.has_max_rows_ != x.has_max_rows_:
            return 0
        if self.has_max_rows_ and self.max_rows_ != x.max_rows_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_queue_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: queue_name not set.')
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += self.lengthString(len(self.queue_name_))
        if self.has_start_task_name_:
            n += 1 + self.lengthString(len(self.start_task_name_))
        if self.has_start_eta_usec_:
            n += 1 + self.lengthVarInt64(self.start_eta_usec_)
        if self.has_start_tag_:
            n += 1 + self.lengthString(len(self.start_tag_))
        if self.has_max_rows_:
            n += 1 + self.lengthVarInt64(self.max_rows_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_start_task_name_:
            n += 1 + self.lengthString(len(self.start_task_name_))
        if self.has_start_eta_usec_:
            n += 1 + self.lengthVarInt64(self.start_eta_usec_)
        if self.has_start_tag_:
            n += 1 + self.lengthString(len(self.start_tag_))
        if self.has_max_rows_:
            n += 1 + self.lengthVarInt64(self.max_rows_)
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_queue_name()
        self.clear_start_task_name()
        self.clear_start_eta_usec()
        self.clear_start_tag()
        self.clear_max_rows()

    def OutputUnchecked(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        out.putVarInt32(18)
        out.putPrefixedString(self.queue_name_)
        if self.has_start_task_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.start_task_name_)
        if self.has_start_eta_usec_:
            out.putVarInt32(32)
            out.putVarInt64(self.start_eta_usec_)
        if self.has_max_rows_:
            out.putVarInt32(40)
            out.putVarInt32(self.max_rows_)
        if self.has_start_tag_:
            out.putVarInt32(50)
            out.putPrefixedString(self.start_tag_)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        if self.has_queue_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_)
        if self.has_start_task_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.start_task_name_)
        if self.has_start_eta_usec_:
            out.putVarInt32(32)
            out.putVarInt64(self.start_eta_usec_)
        if self.has_max_rows_:
            out.putVarInt32(40)
            out.putVarInt32(self.max_rows_)
        if self.has_start_tag_:
            out.putVarInt32(50)
            out.putPrefixedString(self.start_tag_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 26:
                self.set_start_task_name(d.getPrefixedString())
                continue
            if tt == 32:
                self.set_start_eta_usec(d.getVarInt64())
                continue
            if tt == 40:
                self.set_max_rows(d.getVarInt32())
                continue
            if tt == 50:
                self.set_start_tag(d.getPrefixedString())
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
        if self.has_start_task_name_:
            res += prefix + 'start_task_name: %s\n' % self.DebugFormatString(self.start_task_name_)
        if self.has_start_eta_usec_:
            res += prefix + 'start_eta_usec: %s\n' % self.DebugFormatInt64(self.start_eta_usec_)
        if self.has_start_tag_:
            res += prefix + 'start_tag: %s\n' % self.DebugFormatString(self.start_tag_)
        if self.has_max_rows_:
            res += prefix + 'max_rows: %s\n' % self.DebugFormatInt32(self.max_rows_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kqueue_name = 2
    kstart_task_name = 3
    kstart_eta_usec = 4
    kstart_tag = 6
    kmax_rows = 5
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'queue_name', 3: 'start_task_name', 4: 'start_eta_usec', 5: 'max_rows', 6: 'start_tag'}, 6)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.STRING}, 6, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueQueryTasksRequest'