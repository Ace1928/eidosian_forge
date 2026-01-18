import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def _listTestHelper(self, command, listOutput, expectedOutput):
    """
        Exercise handling by the implementation of I{LIST} or I{NLST} of certain
        return values and types from an L{IFTPShell.list} implementation.

        This will issue C{command} and assert that if the L{IFTPShell.list}
        implementation includes C{listOutput} as one of the file entries then
        the result given to the client is matches C{expectedOutput}.

        @param command: Either C{b"LIST"} or C{b"NLST"}
        @type command: L{bytes}

        @param listOutput: A value suitable to be used as an element of the list
            returned by L{IFTPShell.list}.  Vary the values and types of the
            contents to exercise different code paths in the server's handling
            of this result.

        @param expectedOutput: A line of output to expect as a result of
            C{listOutput} being transformed into a response to the command
            issued.
        @type expectedOutput: L{bytes}

        @return: A L{Deferred} which fires when the test is done, either with an
            L{Failure} if the test failed or with a function object if it
            succeeds.  The function object is the function which implements
            L{IFTPShell.list} (and is useful to make assertions about what
            warnings might have been emitted).
        @rtype: L{Deferred}
        """
    d = self._anonymousLogin()

    def patchedList(segments, keys=()):
        return defer.succeed([listOutput])

    def loggedIn(result):
        self.serverProtocol.shell.list = patchedList
        return result
    d.addCallback(loggedIn)
    self._download(f'{command} something', chainDeferred=d)

    def checkDownload(download):
        self.assertEqual(expectedOutput, download)
        return patchedList
    return d.addCallback(checkDownload)