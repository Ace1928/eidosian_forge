from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class CreateLoginURLResponse(ProtocolBuffer.ProtocolMessage):
    has_login_url_ = 0
    login_url_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def login_url(self):
        return self.login_url_

    def set_login_url(self, x):
        self.has_login_url_ = 1
        self.login_url_ = x

    def clear_login_url(self):
        if self.has_login_url_:
            self.has_login_url_ = 0
            self.login_url_ = ''

    def has_login_url(self):
        return self.has_login_url_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_login_url():
            self.set_login_url(x.login_url())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_login_url_ != x.has_login_url_:
            return 0
        if self.has_login_url_ and self.login_url_ != x.login_url_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_login_url_:
            n += 1 + self.lengthString(len(self.login_url_))
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_login_url_:
            n += 1 + self.lengthString(len(self.login_url_))
        return n

    def Clear(self):
        self.clear_login_url()

    def OutputUnchecked(self, out):
        if self.has_login_url_:
            out.putVarInt32(10)
            out.putPrefixedString(self.login_url_)

    def OutputPartial(self, out):
        if self.has_login_url_:
            out.putVarInt32(10)
            out.putPrefixedString(self.login_url_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_login_url(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_login_url_:
            res += prefix + 'login_url: %s\n' % self.DebugFormatString(self.login_url_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    klogin_url = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'login_url'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.CreateLoginURLResponse'