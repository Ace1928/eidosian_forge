from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueAddRequest_CronTimetable(ProtocolBuffer.ProtocolMessage):
    has_schedule_ = 0
    schedule_ = ''
    has_timezone_ = 0
    timezone_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def schedule(self):
        return self.schedule_

    def set_schedule(self, x):
        self.has_schedule_ = 1
        self.schedule_ = x

    def clear_schedule(self):
        if self.has_schedule_:
            self.has_schedule_ = 0
            self.schedule_ = ''

    def has_schedule(self):
        return self.has_schedule_

    def timezone(self):
        return self.timezone_

    def set_timezone(self, x):
        self.has_timezone_ = 1
        self.timezone_ = x

    def clear_timezone(self):
        if self.has_timezone_:
            self.has_timezone_ = 0
            self.timezone_ = ''

    def has_timezone(self):
        return self.has_timezone_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_schedule():
            self.set_schedule(x.schedule())
        if x.has_timezone():
            self.set_timezone(x.timezone())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_schedule_ != x.has_schedule_:
            return 0
        if self.has_schedule_ and self.schedule_ != x.schedule_:
            return 0
        if self.has_timezone_ != x.has_timezone_:
            return 0
        if self.has_timezone_ and self.timezone_ != x.timezone_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_schedule_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: schedule not set.')
        if not self.has_timezone_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: timezone not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.schedule_))
        n += self.lengthString(len(self.timezone_))
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_schedule_:
            n += 1
            n += self.lengthString(len(self.schedule_))
        if self.has_timezone_:
            n += 1
            n += self.lengthString(len(self.timezone_))
        return n

    def Clear(self):
        self.clear_schedule()
        self.clear_timezone()

    def OutputUnchecked(self, out):
        out.putVarInt32(106)
        out.putPrefixedString(self.schedule_)
        out.putVarInt32(114)
        out.putPrefixedString(self.timezone_)

    def OutputPartial(self, out):
        if self.has_schedule_:
            out.putVarInt32(106)
            out.putPrefixedString(self.schedule_)
        if self.has_timezone_:
            out.putVarInt32(114)
            out.putPrefixedString(self.timezone_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 100:
                break
            if tt == 106:
                self.set_schedule(d.getPrefixedString())
                continue
            if tt == 114:
                self.set_timezone(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_schedule_:
            res += prefix + 'schedule: %s\n' % self.DebugFormatString(self.schedule_)
        if self.has_timezone_:
            res += prefix + 'timezone: %s\n' % self.DebugFormatString(self.timezone_)
        return res