import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def _ebChannelRequest(self, result, localChannel):
    """
        Called if the other wisde wanted a reply to the channel requeset and
        the channel request failed.

        @param result: a Failure, but it's not used.
        @param localChannel: the local channel ID of the channel to which the
            request was made.
        @type localChannel: L{int}
        """
    self.transport.sendPacket(MSG_CHANNEL_FAILURE, struct.pack('>L', self.localToRemoteChannel[localChannel]))