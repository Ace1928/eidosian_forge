from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueFetchQueueStatsResponse_QueueStats(ProtocolBuffer.ProtocolMessage):
    has_num_tasks_ = 0
    num_tasks_ = 0
    has_oldest_eta_usec_ = 0
    oldest_eta_usec_ = 0
    has_scanner_info_ = 0
    scanner_info_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def num_tasks(self):
        return self.num_tasks_

    def set_num_tasks(self, x):
        self.has_num_tasks_ = 1
        self.num_tasks_ = x

    def clear_num_tasks(self):
        if self.has_num_tasks_:
            self.has_num_tasks_ = 0
            self.num_tasks_ = 0

    def has_num_tasks(self):
        return self.has_num_tasks_

    def oldest_eta_usec(self):
        return self.oldest_eta_usec_

    def set_oldest_eta_usec(self, x):
        self.has_oldest_eta_usec_ = 1
        self.oldest_eta_usec_ = x

    def clear_oldest_eta_usec(self):
        if self.has_oldest_eta_usec_:
            self.has_oldest_eta_usec_ = 0
            self.oldest_eta_usec_ = 0

    def has_oldest_eta_usec(self):
        return self.has_oldest_eta_usec_

    def scanner_info(self):
        if self.scanner_info_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.scanner_info_ is None:
                    self.scanner_info_ = TaskQueueScannerQueueInfo()
            finally:
                self.lazy_init_lock_.release()
        return self.scanner_info_

    def mutable_scanner_info(self):
        self.has_scanner_info_ = 1
        return self.scanner_info()

    def clear_scanner_info(self):
        if self.has_scanner_info_:
            self.has_scanner_info_ = 0
            if self.scanner_info_ is not None:
                self.scanner_info_.Clear()

    def has_scanner_info(self):
        return self.has_scanner_info_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_num_tasks():
            self.set_num_tasks(x.num_tasks())
        if x.has_oldest_eta_usec():
            self.set_oldest_eta_usec(x.oldest_eta_usec())
        if x.has_scanner_info():
            self.mutable_scanner_info().MergeFrom(x.scanner_info())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_num_tasks_ != x.has_num_tasks_:
            return 0
        if self.has_num_tasks_ and self.num_tasks_ != x.num_tasks_:
            return 0
        if self.has_oldest_eta_usec_ != x.has_oldest_eta_usec_:
            return 0
        if self.has_oldest_eta_usec_ and self.oldest_eta_usec_ != x.oldest_eta_usec_:
            return 0
        if self.has_scanner_info_ != x.has_scanner_info_:
            return 0
        if self.has_scanner_info_ and self.scanner_info_ != x.scanner_info_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_num_tasks_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: num_tasks not set.')
        if not self.has_oldest_eta_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: oldest_eta_usec not set.')
        if self.has_scanner_info_ and (not self.scanner_info_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.num_tasks_)
        n += self.lengthVarInt64(self.oldest_eta_usec_)
        if self.has_scanner_info_:
            n += 1 + self.lengthString(self.scanner_info_.ByteSize())
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_num_tasks_:
            n += 1
            n += self.lengthVarInt64(self.num_tasks_)
        if self.has_oldest_eta_usec_:
            n += 1
            n += self.lengthVarInt64(self.oldest_eta_usec_)
        if self.has_scanner_info_:
            n += 1 + self.lengthString(self.scanner_info_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_num_tasks()
        self.clear_oldest_eta_usec()
        self.clear_scanner_info()

    def OutputUnchecked(self, out):
        out.putVarInt32(16)
        out.putVarInt32(self.num_tasks_)
        out.putVarInt32(24)
        out.putVarInt64(self.oldest_eta_usec_)
        if self.has_scanner_info_:
            out.putVarInt32(34)
            out.putVarInt32(self.scanner_info_.ByteSize())
            self.scanner_info_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_num_tasks_:
            out.putVarInt32(16)
            out.putVarInt32(self.num_tasks_)
        if self.has_oldest_eta_usec_:
            out.putVarInt32(24)
            out.putVarInt64(self.oldest_eta_usec_)
        if self.has_scanner_info_:
            out.putVarInt32(34)
            out.putVarInt32(self.scanner_info_.ByteSizePartial())
            self.scanner_info_.OutputPartial(out)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 16:
                self.set_num_tasks(d.getVarInt32())
                continue
            if tt == 24:
                self.set_oldest_eta_usec(d.getVarInt64())
                continue
            if tt == 34:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_scanner_info().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_num_tasks_:
            res += prefix + 'num_tasks: %s\n' % self.DebugFormatInt32(self.num_tasks_)
        if self.has_oldest_eta_usec_:
            res += prefix + 'oldest_eta_usec: %s\n' % self.DebugFormatInt64(self.oldest_eta_usec_)
        if self.has_scanner_info_:
            res += prefix + 'scanner_info <\n'
            res += self.scanner_info_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res