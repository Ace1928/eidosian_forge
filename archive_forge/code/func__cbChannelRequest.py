import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def _cbChannelRequest(self, result, localChannel):
    """
        Called back if the other side wanted a reply to a channel request.  If
        the result is true, send a MSG_CHANNEL_SUCCESS.  Otherwise, raise
        a C{error.ConchError}

        @param result: the value returned from the channel's requestReceived()
            method.  If it's False, the request failed.
        @type result: L{bool}
        @param localChannel: the local channel ID of the channel to which the
            request was made.
        @type localChannel: L{int}
        @raises ConchError: if the result is False.
        """
    if not result:
        raise error.ConchError('failed request')
    self.transport.sendPacket(MSG_CHANNEL_SUCCESS, struct.pack('>L', self.localToRemoteChannel[localChannel]))