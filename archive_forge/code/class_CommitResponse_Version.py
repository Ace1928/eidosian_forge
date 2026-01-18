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
class CommitResponse_Version(ProtocolBuffer.ProtocolMessage):
    has_root_entity_key_ = 0
    has_version_ = 0
    version_ = 0

    def __init__(self, contents=None):
        self.root_entity_key_ = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.Reference()
        if contents is not None:
            self.MergeFromString(contents)

    def root_entity_key(self):
        return self.root_entity_key_

    def mutable_root_entity_key(self):
        self.has_root_entity_key_ = 1
        return self.root_entity_key_

    def clear_root_entity_key(self):
        self.has_root_entity_key_ = 0
        self.root_entity_key_.Clear()

    def has_root_entity_key(self):
        return self.has_root_entity_key_

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
        if x.has_root_entity_key():
            self.mutable_root_entity_key().MergeFrom(x.root_entity_key())
        if x.has_version():
            self.set_version(x.version())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_root_entity_key_ != x.has_root_entity_key_:
            return 0
        if self.has_root_entity_key_ and self.root_entity_key_ != x.root_entity_key_:
            return 0
        if self.has_version_ != x.has_version_:
            return 0
        if self.has_version_ and self.version_ != x.version_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_root_entity_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: root_entity_key not set.')
        elif not self.root_entity_key_.IsInitialized(debug_strs):
            initialized = 0
        if not self.has_version_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: version not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.root_entity_key_.ByteSize())
        n += self.lengthVarInt64(self.version_)
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_root_entity_key_:
            n += 1
            n += self.lengthString(self.root_entity_key_.ByteSizePartial())
        if self.has_version_:
            n += 1
            n += self.lengthVarInt64(self.version_)
        return n

    def Clear(self):
        self.clear_root_entity_key()
        self.clear_version()

    def OutputUnchecked(self, out):
        out.putVarInt32(34)
        out.putVarInt32(self.root_entity_key_.ByteSize())
        self.root_entity_key_.OutputUnchecked(out)
        out.putVarInt32(40)
        out.putVarInt64(self.version_)

    def OutputPartial(self, out):
        if self.has_root_entity_key_:
            out.putVarInt32(34)
            out.putVarInt32(self.root_entity_key_.ByteSizePartial())
            self.root_entity_key_.OutputPartial(out)
        if self.has_version_:
            out.putVarInt32(40)
            out.putVarInt64(self.version_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 28:
                break
            if tt == 34:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_root_entity_key().TryMerge(tmp)
                continue
            if tt == 40:
                self.set_version(d.getVarInt64())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_root_entity_key_:
            res += prefix + 'root_entity_key <\n'
            res += self.root_entity_key_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_version_:
            res += prefix + 'version: %s\n' % self.DebugFormatInt64(self.version_)
        return res