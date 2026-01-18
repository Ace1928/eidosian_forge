from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueBulkAddResponse_TaskResult(ProtocolBuffer.ProtocolMessage):
    has_result_ = 0
    result_ = 0
    has_chosen_task_name_ = 0
    chosen_task_name_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def result(self):
        return self.result_

    def set_result(self, x):
        self.has_result_ = 1
        self.result_ = x

    def clear_result(self):
        if self.has_result_:
            self.has_result_ = 0
            self.result_ = 0

    def has_result(self):
        return self.has_result_

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
        if x.has_result():
            self.set_result(x.result())
        if x.has_chosen_task_name():
            self.set_chosen_task_name(x.chosen_task_name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_result_ != x.has_result_:
            return 0
        if self.has_result_ and self.result_ != x.result_:
            return 0
        if self.has_chosen_task_name_ != x.has_chosen_task_name_:
            return 0
        if self.has_chosen_task_name_ and self.chosen_task_name_ != x.chosen_task_name_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_result_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: result not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.result_)
        if self.has_chosen_task_name_:
            n += 1 + self.lengthString(len(self.chosen_task_name_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_result_:
            n += 1
            n += self.lengthVarInt64(self.result_)
        if self.has_chosen_task_name_:
            n += 1 + self.lengthString(len(self.chosen_task_name_))
        return n

    def Clear(self):
        self.clear_result()
        self.clear_chosen_task_name()

    def OutputUnchecked(self, out):
        out.putVarInt32(16)
        out.putVarInt32(self.result_)
        if self.has_chosen_task_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.chosen_task_name_)

    def OutputPartial(self, out):
        if self.has_result_:
            out.putVarInt32(16)
            out.putVarInt32(self.result_)
        if self.has_chosen_task_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.chosen_task_name_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 16:
                self.set_result(d.getVarInt32())
                continue
            if tt == 26:
                self.set_chosen_task_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_result_:
            res += prefix + 'result: %s\n' % self.DebugFormatInt32(self.result_)
        if self.has_chosen_task_name_:
            res += prefix + 'chosen_task_name: %s\n' % self.DebugFormatString(self.chosen_task_name_)
        return res