from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueAddResponse(ProtocolBuffer.ProtocolMessage):
    has_chosen_task_name_ = 0
    chosen_task_name_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def chosen_task_name(self):
        return self.chosen_task_name_

    def set_chosen_task_name(self, x):
        self.has_chosen_task_name_ = 1
        self.chosen_task_name_ = x

    def clear_chosen_task_name(self):
        if self.has_chosen_task_name_:
            self.has_chosen_task_name_ = 0
            self.chosen_task_name_ = ''

    def has_chosen_task_name(self):
        return self.has_chosen_task_name_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_chosen_task_name():
            self.set_chosen_task_name(x.chosen_task_name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_chosen_task_name_ != x.has_chosen_task_name_:
            return 0
        if self.has_chosen_task_name_ and self.chosen_task_name_ != x.chosen_task_name_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_chosen_task_name_:
            n += 1 + self.lengthString(len(self.chosen_task_name_))
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_chosen_task_name_:
            n += 1 + self.lengthString(len(self.chosen_task_name_))
        return n

    def Clear(self):
        self.clear_chosen_task_name()

    def OutputUnchecked(self, out):
        if self.has_chosen_task_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.chosen_task_name_)

    def OutputPartial(self, out):
        if self.has_chosen_task_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.chosen_task_name_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_chosen_task_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_chosen_task_name_:
            res += prefix + 'chosen_task_name: %s\n' % self.DebugFormatString(self.chosen_task_name_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kchosen_task_name = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'chosen_task_name'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueAddResponse'