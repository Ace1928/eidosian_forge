from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
class CompiledCursor(ProtocolBuffer.ProtocolMessage):
    has_position_ = 0
    position_ = None
    has_postfix_position_ = 0
    postfix_position_ = None
    has_absolute_position_ = 0
    absolute_position_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def position(self):
        if self.position_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.position_ is None:
                    self.position_ = CompiledCursor_Position()
            finally:
                self.lazy_init_lock_.release()
        return self.position_

    def mutable_position(self):
        self.has_position_ = 1
        return self.position()

    def clear_position(self):
        if self.has_position_:
            self.has_position_ = 0
            if self.position_ is not None:
                self.position_.Clear()

    def has_position(self):
        return self.has_position_

    def postfix_position(self):
        if self.postfix_position_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.postfix_position_ is None:
                    self.postfix_position_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.IndexPostfix()
            finally:
                self.lazy_init_lock_.release()
        return self.postfix_position_

    def mutable_postfix_position(self):
        self.has_postfix_position_ = 1
        return self.postfix_position()

    def clear_postfix_position(self):
        if self.has_postfix_position_:
            self.has_postfix_position_ = 0
            if self.postfix_position_ is not None:
                self.postfix_position_.Clear()

    def has_postfix_position(self):
        return self.has_postfix_position_

    def absolute_position(self):
        if self.absolute_position_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.absolute_position_ is None:
                    self.absolute_position_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.IndexPosition()
            finally:
                self.lazy_init_lock_.release()
        return self.absolute_position_

    def mutable_absolute_position(self):
        self.has_absolute_position_ = 1
        return self.absolute_position()

    def clear_absolute_position(self):
        if self.has_absolute_position_:
            self.has_absolute_position_ = 0
            if self.absolute_position_ is not None:
                self.absolute_position_.Clear()

    def has_absolute_position(self):
        return self.has_absolute_position_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_position():
            self.mutable_position().MergeFrom(x.position())
        if x.has_postfix_position():
            self.mutable_postfix_position().MergeFrom(x.postfix_position())
        if x.has_absolute_position():
            self.mutable_absolute_position().MergeFrom(x.absolute_position())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_position_ != x.has_position_:
            return 0
        if self.has_position_ and self.position_ != x.position_:
            return 0
        if self.has_postfix_position_ != x.has_postfix_position_:
            return 0
        if self.has_postfix_position_ and self.postfix_position_ != x.postfix_position_:
            return 0
        if self.has_absolute_position_ != x.has_absolute_position_:
            return 0
        if self.has_absolute_position_ and self.absolute_position_ != x.absolute_position_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_position_ and (not self.position_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_postfix_position_ and (not self.postfix_position_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_absolute_position_ and (not self.absolute_position_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_position_:
            n += 2 + self.position_.ByteSize()
        if self.has_postfix_position_:
            n += 1 + self.lengthString(self.postfix_position_.ByteSize())
        if self.has_absolute_position_:
            n += 1 + self.lengthString(self.absolute_position_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_position_:
            n += 2 + self.position_.ByteSizePartial()
        if self.has_postfix_position_:
            n += 1 + self.lengthString(self.postfix_position_.ByteSizePartial())
        if self.has_absolute_position_:
            n += 1 + self.lengthString(self.absolute_position_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_position()
        self.clear_postfix_position()
        self.clear_absolute_position()

    def OutputUnchecked(self, out):
        if self.has_postfix_position_:
            out.putVarInt32(10)
            out.putVarInt32(self.postfix_position_.ByteSize())
            self.postfix_position_.OutputUnchecked(out)
        if self.has_position_:
            out.putVarInt32(19)
            self.position_.OutputUnchecked(out)
            out.putVarInt32(20)
        if self.has_absolute_position_:
            out.putVarInt32(26)
            out.putVarInt32(self.absolute_position_.ByteSize())
            self.absolute_position_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_postfix_position_:
            out.putVarInt32(10)
            out.putVarInt32(self.postfix_position_.ByteSizePartial())
            self.postfix_position_.OutputPartial(out)
        if self.has_position_:
            out.putVarInt32(19)
            self.position_.OutputPartial(out)
            out.putVarInt32(20)
        if self.has_absolute_position_:
            out.putVarInt32(26)
            out.putVarInt32(self.absolute_position_.ByteSizePartial())
            self.absolute_position_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_postfix_position().TryMerge(tmp)
                continue
            if tt == 19:
                self.mutable_position().TryMerge(d)
                continue
            if tt == 26:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_absolute_position().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_position_:
            res += prefix + 'Position {\n'
            res += self.position_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_postfix_position_:
            res += prefix + 'postfix_position <\n'
            res += self.postfix_position_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_absolute_position_:
            res += prefix + 'absolute_position <\n'
            res += self.absolute_position_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kPositionGroup = 2
    kPositionstart_key = 27
    kPositionIndexValueGroup = 29
    kPositionIndexValueproperty = 30
    kPositionIndexValuevalue = 31
    kPositionkey = 32
    kPositionstart_inclusive = 28
    kPositionbefore_ascending = 33
    kpostfix_position = 1
    kabsolute_position = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'postfix_position', 2: 'Position', 3: 'absolute_position', 27: 'start_key', 28: 'start_inclusive', 29: 'IndexValue', 30: 'property', 31: 'value', 32: 'key', 33: 'before_ascending'}, 33)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STARTGROUP, 3: ProtocolBuffer.Encoder.STRING, 27: ProtocolBuffer.Encoder.STRING, 28: ProtocolBuffer.Encoder.NUMERIC, 29: ProtocolBuffer.Encoder.STARTGROUP, 30: ProtocolBuffer.Encoder.STRING, 31: ProtocolBuffer.Encoder.STRING, 32: ProtocolBuffer.Encoder.STRING, 33: ProtocolBuffer.Encoder.NUMERIC}, 33, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.CompiledCursor'