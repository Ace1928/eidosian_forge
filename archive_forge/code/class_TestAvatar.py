import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
class TestAvatar:
    """
    A mocked-up version of twisted.conch.avatar.ConchUser
    """
    _ARGS_ERROR_CODE = 123

    def lookupChannel(self, channelType, windowSize, maxPacket, data):
        """
        The server wants us to return a channel.  If the requested channel is
        our TestChannel, return it, otherwise return None.
        """
        if channelType == TestChannel.name:
            return TestChannel(remoteWindow=windowSize, remoteMaxPacket=maxPacket, data=data, avatar=self)
        elif channelType == b'conch-error-args':
            raise error.ConchError(self._ARGS_ERROR_CODE, 'error args in wrong order')

    def gotGlobalRequest(self, requestType, data):
        """
        The client has made a global request.  If the global request is
        'TestGlobal', return True.  If the global request is 'TestData',
        return True and the request-specific data we received.  Otherwise,
        return False.
        """
        if requestType == b'TestGlobal':
            return True
        elif requestType == b'TestData':
            return (True, data)
        else:
            return False