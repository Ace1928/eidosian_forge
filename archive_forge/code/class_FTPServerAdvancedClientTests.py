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
class FTPServerAdvancedClientTests(FTPServerTestCase):
    """
    Test FTP server with the L{ftp.FTPClient} class.
    """
    clientFactory = ftp.FTPClient

    def test_anonymousSTOR(self):
        """
        Try to make an STOR as anonymous, and check that we got a permission
        denied error.
        """

        def eb(res):
            res.trap(ftp.CommandFailed)
            self.assertEqual(res.value.args[0][0], '550 foo: Permission denied.')
        d1, d2 = self.client.storeFile('foo')
        d2.addErrback(eb)
        return defer.gatherResults([d1, d2])

    def test_STORtransferErrorIsReturned(self):
        """
        Any FTP error raised by STOR while transferring the file is returned
        to the client.
        """

        class FailingFileWriter(ftp._FileWriter):

            def receive(self):
                return defer.fail(ftp.IsADirectoryError('failing_file'))

        def failingSTOR(a, b):
            return defer.succeed(FailingFileWriter(None))
        self.patch(ftp.FTPAnonymousShell, 'openForWriting', failingSTOR)

        def eb(res):
            res.trap(ftp.CommandFailed)
            logs = self.flushLoggedErrors()
            self.assertEqual(1, len(logs))
            self.assertIsInstance(logs[0].value, ftp.IsADirectoryError)
            self.assertEqual(res.value.args[0][0], '550 failing_file: is a directory')
        d1, d2 = self.client.storeFile('failing_file')
        d2.addErrback(eb)
        return defer.gatherResults([d1, d2])

    def test_STORunknownTransferErrorBecomesAbort(self):
        """
        Any non FTP error raised by STOR while transferring the file is
        converted into a critical error and transfer is closed.

        The unknown error is logged.
        """

        class FailingFileWriter(ftp._FileWriter):

            def receive(self):
                return defer.fail(AssertionError())

        def failingSTOR(a, b):
            return defer.succeed(FailingFileWriter(None))
        self.patch(ftp.FTPAnonymousShell, 'openForWriting', failingSTOR)

        def eb(res):
            res.trap(ftp.CommandFailed)
            logs = self.flushLoggedErrors()
            self.assertEqual(1, len(logs))
            self.assertIsInstance(logs[0].value, AssertionError)
            self.assertEqual(res.value.args[0][0], '426 Transfer aborted.  Data connection closed.')
        d1, d2 = self.client.storeFile('failing_file')
        d2.addErrback(eb)
        return defer.gatherResults([d1, d2])

    def test_RETRreadError(self):
        """
        Any errors during reading a file inside a RETR should be returned to
        the client.
        """

        class FailingFileReader(ftp._FileReader):

            def send(self, consumer):
                return defer.fail(ftp.IsADirectoryError('blah'))

        def failingRETR(a, b):
            return defer.succeed(FailingFileReader(None))
        self.patch(ftp.FTPAnonymousShell, 'openForReading', failingRETR)

        def check_response(failure):
            self.flushLoggedErrors()
            failure.trap(ftp.CommandFailed)
            self.assertEqual(failure.value.args[0][0], '125 Data connection already open, starting transfer')
            self.assertEqual(failure.value.args[0][1], '550 blah: is a directory')
        proto = _BufferingProtocol()
        d = self.client.retrieveFile('failing_file', proto)
        d.addErrback(check_response)
        return d