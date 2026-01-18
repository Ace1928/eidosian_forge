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
class GetResponse_Entity(ProtocolBuffer.ProtocolMessage):
    has_entity_ = 0
    entity_ = None
    has_key_ = 0
    key_ = None
    has_version_ = 0
    version_ = 0

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def entity(self):
        if self.entity_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.entity_ is None:
                    self.entity_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.EntityProto()
            finally:
                self.lazy_init_lock_.release()
        return self.entity_

    def mutable_entity(self):
        self.has_entity_ = 1
        return self.entity()

    def clear_entity(self):
        if self.has_entity_:
            self.has_entity_ = 0
            if self.entity_ is not None:
                self.entity_.Clear()

    def has_entity(self):
        return self.has_entity_

    def key(self):
        if self.key_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.key_ is None:
                    self.key_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
            finally:
                self.lazy_init_lock_.release()
        return self.key_

    def mutable_key(self):
        self.has_key_ = 1
        return self.key()

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            if self.key_ is not None:
                self.key_.Clear()

    def has_key(self):
        return self.has_key_

    def version(self):
        return self.version_

    def set_version(self, x):
        self.has_version_ = 1
        self.version_ = x

    def clear_version(self):
        if self.has_version_:
            self.has_version_ = 0
            self.version_ = 0

    def has_version(self):
        return self.has_version_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_entity():
            self.mutable_entity().MergeFrom(x.entity())
        if x.has_key():
            self.mutable_key().MergeFrom(x.key())
        if x.has_version():
            self.set_version(x.version())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_entity_ != x.has_entity_:
            return 0
        if self.has_entity_ and self.entity_ != x.entity_:
            return 0
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_version_ != x.has_version_:
            return 0
        if self.has_version_ and self.version_ != x.version_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_entity_ and (not self.entity_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_key_ and (not self.key_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_entity_:
            n += 1 + self.lengthString(self.entity_.ByteSize())
        if self.has_key_:
            n += 1 + self.lengthString(self.key_.ByteSize())
        if self.has_version_:
            n += 1 + self.lengthVarInt64(self.version_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_entity_:
            n += 1 + self.lengthString(self.entity_.ByteSizePartial())
        if self.has_key_:
            n += 1 + self.lengthString(self.key_.ByteSizePartial())
        if self.has_version_:
            n += 1 + self.lengthVarInt64(self.version_)
        return n

    def Clear(self):
        self.clear_entity()
        self.clear_key()
        self.clear_version()

    def OutputUnchecked(self, out):
        if self.has_entity_:
            out.putVarInt32(18)
            out.putVarInt32(self.entity_.ByteSize())
            self.entity_.OutputUnchecked(out)
        if self.has_version_:
            out.putVarInt32(24)
            out.putVarInt64(self.version_)
        if self.has_key_:
            out.putVarInt32(34)
            out.putVarInt32(self.key_.ByteSize())
            self.key_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_entity_:
            out.putVarInt32(18)
            out.putVarInt32(self.entity_.ByteSizePartial())
            self.entity_.OutputPartial(out)
        if self.has_version_:
            out.putVarInt32(24)
            out.putVarInt64(self.version_)
        if self.has_key_:
            out.putVarInt32(34)
            out.putVarInt32(self.key_.ByteSizePartial())
            self.key_.OutputPartial(out)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_entity().TryMerge(tmp)
                continue
            if tt == 24:
                self.set_version(d.getVarInt64())
                continue
            if tt == 34:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_key().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_entity_:
            res += prefix + 'entity <\n'
            res += self.entity_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_key_:
            res += prefix + 'key <\n'
            res += self.key_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_version_:
            res += prefix + 'version: %s\n' % self.DebugFormatInt64(self.version_)
        return res