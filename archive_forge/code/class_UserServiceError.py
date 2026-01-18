from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class UserServiceError(ProtocolBuffer.ProtocolMessage):
    OK = 0
    REDIRECT_URL_TOO_LONG = 1
    NOT_ALLOWED = 2
    OAUTH_INVALID_TOKEN = 3
    OAUTH_INVALID_REQUEST = 4
    OAUTH_ERROR = 5
    _ErrorCode_NAMES = {0: 'OK', 1: 'REDIRECT_URL_TOO_LONG', 2: 'NOT_ALLOWED', 3: 'OAUTH_INVALID_TOKEN', 4: 'OAUTH_INVALID_REQUEST', 5: 'OAUTH_ERROR'}

    def ErrorCode_Name(cls, x):
        return cls._ErrorCode_NAMES.get(x, '')
    ErrorCode_Name = classmethod(ErrorCode_Name)

    def __init__(self, contents=None):
        pass
        if contents is not None:
            self.MergeFromString(contents)

    def MergeFrom(self, x):
        assert x is not self

    def Equals(self, x):
        if x is self:
            return 1
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        return n

    def ByteSizePartial(self):
        n = 0
        return n

    def Clear(self):
        pass

    def OutputUnchecked(self, out):
        pass

    def OutputPartial(self, out):
        pass

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode'}, 0)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC}, 0, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.UserServiceError'