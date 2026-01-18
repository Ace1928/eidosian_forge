import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
class _PTYPath:
    """
    A L{FilePath}-like object which can be opened to create a L{_ReadFile} with
    certain contents.
    """

    def __init__(self, contents):
        """
        @param contents: L{bytes} which will be the contents of the
            L{_ReadFile} this path can open.
        """
        self.contents = contents

    def open(self, mode):
        """
        If the mode is r+, return a L{_ReadFile} with the contents given to
        this path's initializer.

        @raise OSError: If the mode is unsupported.

        @return: A L{_ReadFile} instance
        """
        if mode == 'rb+':
            return _ReadFile(self.contents)
        raise OSError(ENOSYS, 'Function not implemented')