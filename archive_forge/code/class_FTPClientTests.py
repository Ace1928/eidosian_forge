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
class FTPClientTests(TestCase):
    """
    Test advanced FTP client commands.
    """

    def setUp(self):
        """
        Create a FTP client and connect it to fake transport.
        """
        self.client = ftp.FTPClient()
        self.transport = proto_helpers.StringTransportWithDisconnection()
        self.client.makeConnection(self.transport)
        self.transport.protocol = self.client

    def tearDown(self):
        """
        Deliver disconnection notification to the client so that it can
        perform any cleanup which may be required.
        """
        self.client.connectionLost(error.ConnectionLost())

    def _testLogin(self):
        """
        Test the login part.
        """
        self.assertEqual(self.transport.value(), b'')
        self.client.lineReceived(b'331 Guest login ok, type your email address as password.')
        self.assertEqual(self.transport.value(), b'USER anonymous\r\n')
        self.transport.clear()
        self.client.lineReceived(b'230 Anonymous login ok, access restrictions apply.')
        self.assertEqual(self.transport.value(), b'TYPE I\r\n')
        self.transport.clear()
        self.client.lineReceived(b'200 Type set to I.')

    def test_sendLine(self):
        """
        Test encoding behaviour of sendLine
        """
        self.assertEqual(self.transport.value(), b'')
        self.client.sendLine(None)
        self.assertEqual(self.transport.value(), b'')
        self.client.sendLine('')
        self.assertEqual(self.transport.value(), b'\r\n')
        self.transport.clear()
        self.client.sendLine('é')
        self.assertEqual(self.transport.value(), b'\xe9\r\n')

    def test_CDUP(self):
        """
        Test the CDUP command.

        L{ftp.FTPClient.cdup} should return a Deferred which fires with a
        sequence of one element which is the string the server sent
        indicating that the command was executed successfully.

        (XXX - This is a bad API)
        """

        def cbCdup(res):
            self.assertEqual(res[0], '250 Requested File Action Completed OK')
        self._testLogin()
        d = self.client.cdup().addCallback(cbCdup)
        self.assertEqual(self.transport.value(), b'CDUP\r\n')
        self.transport.clear()
        self.client.lineReceived(b'250 Requested File Action Completed OK')
        return d

    def test_failedCDUP(self):
        """
        Test L{ftp.FTPClient.cdup}'s handling of a failed CDUP command.

        When the CDUP command fails, the returned Deferred should errback
        with L{ftp.CommandFailed}.
        """
        self._testLogin()
        d = self.client.cdup()
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'CDUP\r\n')
        self.transport.clear()
        self.client.lineReceived(b'550 ..: No such file or directory')
        return d

    def test_PWD(self):
        """
        Test the PWD command.

        L{ftp.FTPClient.pwd} should return a Deferred which fires with a
        sequence of one element which is a string representing the current
        working directory on the server.

        (XXX - This is a bad API)
        """

        def cbPwd(res):
            self.assertEqual(ftp.parsePWDResponse(res[0]), '/bar/baz')
        self._testLogin()
        d = self.client.pwd().addCallback(cbPwd)
        self.assertEqual(self.transport.value(), b'PWD\r\n')
        self.client.lineReceived(b'257 "/bar/baz"')
        return d

    def test_failedPWD(self):
        """
        Test a failure in PWD command.

        When the PWD command fails, the returned Deferred should errback
        with L{ftp.CommandFailed}.
        """
        self._testLogin()
        d = self.client.pwd()
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PWD\r\n')
        self.client.lineReceived(b'550 /bar/baz: No such file or directory')
        return d

    def test_CWD(self):
        """
        Test the CWD command.

        L{ftp.FTPClient.cwd} should return a Deferred which fires with a
        sequence of one element which is the string the server sent
        indicating that the command was executed successfully.

        (XXX - This is a bad API)
        """

        def cbCwd(res):
            self.assertEqual(res[0], '250 Requested File Action Completed OK')
        self._testLogin()
        d = self.client.cwd('bar/foo').addCallback(cbCwd)
        self.assertEqual(self.transport.value(), b'CWD bar/foo\r\n')
        self.client.lineReceived(b'250 Requested File Action Completed OK')
        return d

    def test_failedCWD(self):
        """
        Test a failure in CWD command.

        When the PWD command fails, the returned Deferred should errback
        with L{ftp.CommandFailed}.
        """
        self._testLogin()
        d = self.client.cwd('bar/foo')
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'CWD bar/foo\r\n')
        self.client.lineReceived(b'550 bar/foo: No such file or directory')
        return d

    def test_passiveRETR(self):
        """
        Test the RETR command in passive mode: get a file and verify its
        content.

        L{ftp.FTPClient.retrieveFile} should return a Deferred which fires
        with the protocol instance passed to it after the download has
        completed.

        (XXX - This API should be based on producers and consumers)
        """

        def cbRetr(res, proto):
            self.assertEqual(proto.buffer, b'x' * 1000)

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            proto.dataReceived(b'x' * 1000)
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        proto = _BufferingProtocol()
        d = self.client.retrieveFile('spam', proto)
        d.addCallback(cbRetr, proto)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'RETR spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_RETR(self):
        """
        Test the RETR command in non-passive mode.

        Like L{test_passiveRETR} but in the configuration where the server
        establishes the data connection to the client, rather than the other
        way around.
        """
        self.client.passive = False

        def generatePort(portCmd):
            portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
            portCmd.protocol.makeConnection(proto_helpers.StringTransport())
            portCmd.protocol.dataReceived(b'x' * 1000)
            portCmd.protocol.connectionLost(failure.Failure(error.ConnectionDone('')))

        def cbRetr(res, proto):
            self.assertEqual(proto.buffer, b'x' * 1000)
        self.client.generatePortCommand = generatePort
        self._testLogin()
        proto = _BufferingProtocol()
        d = self.client.retrieveFile('spam', proto)
        d.addCallback(cbRetr, proto)
        self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
        self.transport.clear()
        self.client.lineReceived(b'200 PORT OK')
        self.assertEqual(self.transport.value(), b'RETR spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_failedRETR(self):
        """
        Try to RETR an unexisting file.

        L{ftp.FTPClient.retrieveFile} should return a Deferred which
        errbacks with L{ftp.CommandFailed} if the server indicates the file
        cannot be transferred for some reason.
        """

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        proto = _BufferingProtocol()
        d = self.client.retrieveFile('spam', proto)
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'RETR spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'550 spam: No such file or directory')
        return d

    def test_lostRETR(self):
        """
        Try a RETR, but disconnect during the transfer.
        L{ftp.FTPClient.retrieveFile} should return a Deferred which
        errbacks with L{ftp.ConnectionLost)
        """
        self.client.passive = False
        l = []

        def generatePort(portCmd):
            portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
            tr = proto_helpers.StringTransportWithDisconnection()
            portCmd.protocol.makeConnection(tr)
            tr.protocol = portCmd.protocol
            portCmd.protocol.dataReceived(b'x' * 500)
            l.append(tr)
        self.client.generatePortCommand = generatePort
        self._testLogin()
        proto = _BufferingProtocol()
        d = self.client.retrieveFile('spam', proto)
        self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
        self.transport.clear()
        self.client.lineReceived(b'200 PORT OK')
        self.assertEqual(self.transport.value(), b'RETR spam\r\n')
        self.assertTrue(l)
        l[0].loseConnection()
        self.transport.loseConnection()
        self.assertFailure(d, ftp.ConnectionLost)
        return d

    def test_passiveSTOR(self):
        """
        Test the STOR command: send a file and verify its content.

        L{ftp.FTPClient.storeFile} should return a two-tuple of Deferreds.
        The first of which should fire with a protocol instance when the
        data connection has been established and is responsible for sending
        the contents of the file.  The second of which should fire when the
        upload has completed, the data connection has been closed, and the
        server has acknowledged receipt of the file.

        (XXX - storeFile should take a producer as an argument, instead, and
        only return a Deferred which fires when the upload has succeeded or
        failed).
        """
        tr = proto_helpers.StringTransport()

        def cbStore(sender):
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            sender.transport.write(b'x' * 1000)
            sender.finish()
            sender.connectionLost(failure.Failure(error.ConnectionDone('')))

        def cbFinish(ign):
            self.assertEqual(tr.value(), b'x' * 1000)

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(tr)
        self.client.connectFactory = cbConnect
        self._testLogin()
        d1, d2 = self.client.storeFile('spam')
        d1.addCallback(cbStore)
        d2.addCallback(cbFinish)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'STOR spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'226 Transfer Complete.')
        return defer.gatherResults([d1, d2])

    def test_failedSTOR(self):
        """
        Test a failure in the STOR command.

        If the server does not acknowledge successful receipt of the
        uploaded file, the second Deferred returned by
        L{ftp.FTPClient.storeFile} should errback with L{ftp.CommandFailed}.
        """
        tr = proto_helpers.StringTransport()

        def cbStore(sender):
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            sender.transport.write(b'x' * 1000)
            sender.finish()
            sender.connectionLost(failure.Failure(error.ConnectionDone('')))

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(tr)
        self.client.connectFactory = cbConnect
        self._testLogin()
        d1, d2 = self.client.storeFile('spam')
        d1.addCallback(cbStore)
        self.assertFailure(d2, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'STOR spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'426 Transfer aborted.  Data connection closed.')
        return defer.gatherResults([d1, d2])

    def test_STOR(self):
        """
        Test the STOR command in non-passive mode.

        Like L{test_passiveSTOR} but in the configuration where the server
        establishes the data connection to the client, rather than the other
        way around.
        """
        tr = proto_helpers.StringTransport()
        self.client.passive = False

        def generatePort(portCmd):
            portCmd.text = 'PORT ' + ftp.encodeHostPort('127.0.0.1', 9876)
            portCmd.protocol.makeConnection(tr)

        def cbStore(sender):
            self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
            self.transport.clear()
            self.client.lineReceived(b'200 PORT OK')
            self.assertEqual(self.transport.value(), b'STOR spam\r\n')
            self.transport.clear()
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            sender.transport.write(b'x' * 1000)
            sender.finish()
            sender.connectionLost(failure.Failure(error.ConnectionDone('')))
            self.client.lineReceived(b'226 Transfer Complete.')

        def cbFinish(ign):
            self.assertEqual(tr.value(), b'x' * 1000)
        self.client.generatePortCommand = generatePort
        self._testLogin()
        d1, d2 = self.client.storeFile('spam')
        d1.addCallback(cbStore)
        d2.addCallback(cbFinish)
        return defer.gatherResults([d1, d2])

    def test_passiveLIST(self):
        """
        Test the LIST command.

        L{ftp.FTPClient.list} should return a Deferred which fires with a
        protocol instance which was passed to list after the command has
        succeeded.

        (XXX - This is a very unfortunate API; if my understanding is
        correct, the results are always at least line-oriented, so allowing
        a per-line parser function to be specified would make this simpler,
        but a default implementation should really be provided which knows
        how to deal with all the formats used in real servers, so
        application developers never have to care about this insanity.  It
        would also be nice to either get back a Deferred of a list of
        filenames or to be able to consume the files as they are received
        (which the current API does allow, but in a somewhat inconvenient
        fashion) -exarkun)
        """

        def cbList(res, fileList):
            fls = [f['filename'] for f in fileList.files]
            expected = ['foo', 'bar', 'baz']
            expected.sort()
            fls.sort()
            self.assertEqual(fls, expected)

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            sending = [b'-rw-r--r--    0 spam      egg      100 Oct 10 2006 foo\r\n', b'-rw-r--r--    3 spam      egg      100 Oct 10 2006 bar\r\n', b'-rw-r--r--    4 spam      egg      100 Oct 10 2006 baz\r\n']
            for i in sending:
                proto.dataReceived(i)
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        fileList = ftp.FTPFileListProtocol()
        d = self.client.list('foo/bar', fileList).addCallback(cbList, fileList)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'LIST foo/bar\r\n')
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_LIST(self):
        """
        Test the LIST command in non-passive mode.

        Like L{test_passiveLIST} but in the configuration where the server
        establishes the data connection to the client, rather than the other
        way around.
        """
        self.client.passive = False

        def generatePort(portCmd):
            portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
            portCmd.protocol.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            sending = [b'-rw-r--r--    0 spam      egg      100 Oct 10 2006 foo\r\n', b'-rw-r--r--    3 spam      egg      100 Oct 10 2006 bar\r\n', b'-rw-r--r--    4 spam      egg      100 Oct 10 2006 baz\r\n']
            for i in sending:
                portCmd.protocol.dataReceived(i)
            portCmd.protocol.connectionLost(failure.Failure(error.ConnectionDone('')))

        def cbList(res, fileList):
            fls = [f['filename'] for f in fileList.files]
            expected = ['foo', 'bar', 'baz']
            expected.sort()
            fls.sort()
            self.assertEqual(fls, expected)
        self.client.generatePortCommand = generatePort
        self._testLogin()
        fileList = ftp.FTPFileListProtocol()
        d = self.client.list('foo/bar', fileList).addCallback(cbList, fileList)
        self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
        self.transport.clear()
        self.client.lineReceived(b'200 PORT OK')
        self.assertEqual(self.transport.value(), b'LIST foo/bar\r\n')
        self.transport.clear()
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_failedLIST(self):
        """
        Test a failure in LIST command.

        L{ftp.FTPClient.list} should return a Deferred which fails with
        L{ftp.CommandFailed} if the server indicates the indicated path is
        invalid for some reason.
        """

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        fileList = ftp.FTPFileListProtocol()
        d = self.client.list('foo/bar', fileList)
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'LIST foo/bar\r\n')
        self.client.lineReceived(b'550 foo/bar: No such file or directory')
        return d

    def test_NLST(self):
        """
        Test the NLST command in non-passive mode.

        L{ftp.FTPClient.nlst} should return a Deferred which fires with a
        list of filenames when the list command has completed.
        """
        self.client.passive = False

        def generatePort(portCmd):
            portCmd.text = 'PORT {}'.format(ftp.encodeHostPort('127.0.0.1', 9876))
            portCmd.protocol.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            portCmd.protocol.dataReceived(b'foo\r\n')
            portCmd.protocol.dataReceived(b'bar\r\n')
            portCmd.protocol.dataReceived(b'baz\r\n')
            portCmd.protocol.connectionLost(failure.Failure(error.ConnectionDone('')))

        def cbList(res, proto):
            fls = proto.buffer.decode(self.client._encoding).splitlines()
            expected = ['foo', 'bar', 'baz']
            expected.sort()
            fls.sort()
            self.assertEqual(fls, expected)
        self.client.generatePortCommand = generatePort
        self._testLogin()
        lstproto = _BufferingProtocol()
        d = self.client.nlst('foo/bar', lstproto).addCallback(cbList, lstproto)
        self.assertEqual(self.transport.value(), 'PORT {}\r\n'.format(ftp.encodeHostPort('127.0.0.1', 9876)).encode(self.client._encoding))
        self.transport.clear()
        self.client.lineReceived(b'200 PORT OK')
        self.assertEqual(self.transport.value(), b'NLST foo/bar\r\n')
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_passiveNLST(self):
        """
        Test the NLST command.

        Like L{test_passiveNLST} but in the configuration where the server
        establishes the data connection to the client, rather than the other
        way around.
        """

        def cbList(res, proto):
            fls = proto.buffer.splitlines()
            expected = [b'foo', b'bar', b'baz']
            expected.sort()
            fls.sort()
            self.assertEqual(fls, expected)

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(proto_helpers.StringTransport())
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            proto.dataReceived(b'foo\r\n')
            proto.dataReceived(b'bar\r\n')
            proto.dataReceived(b'baz\r\n')
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        lstproto = _BufferingProtocol()
        d = self.client.nlst('foo/bar', lstproto).addCallback(cbList, lstproto)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'NLST foo/bar\r\n')
        self.client.lineReceived(b'226 Transfer Complete.')
        return d

    def test_failedNLST(self):
        """
        Test a failure in NLST command.

        L{ftp.FTPClient.nlst} should return a Deferred which fails with
        L{ftp.CommandFailed} if the server indicates the indicated path is
        invalid for some reason.
        """
        tr = proto_helpers.StringTransport()

        def cbConnect(host, port, factory):
            self.assertEqual(host, '127.0.0.1')
            self.assertEqual(port, 12345)
            proto = factory.buildProtocol((host, port))
            proto.makeConnection(tr)
            self.client.lineReceived(b'150 File status okay; about to open data connection.')
            proto.connectionLost(failure.Failure(error.ConnectionDone('')))
        self.client.connectFactory = cbConnect
        self._testLogin()
        lstproto = _BufferingProtocol()
        d = self.client.nlst('foo/bar', lstproto)
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PASV\r\n')
        self.transport.clear()
        self.client.lineReceived(passivemode_msg(self.client))
        self.assertEqual(self.transport.value(), b'NLST foo/bar\r\n')
        self.client.lineReceived(b'550 foo/bar: No such file or directory')
        return d

    def test_renameFromTo(self):
        """
        L{ftp.FTPClient.rename} issues I{RNTO} and I{RNFR} commands and returns
        a L{Deferred} which fires when a file has successfully been renamed.
        """
        self._testLogin()
        d = self.client.rename('/spam', '/ham')
        self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
        self.transport.clear()
        fromResponse = '350 Requested file action pending further information.\r\n'
        self.client.lineReceived(fromResponse.encode(self.client._encoding))
        self.assertEqual(self.transport.value(), b'RNTO /ham\r\n')
        toResponse = '250 Requested File Action Completed OK'
        self.client.lineReceived(toResponse.encode(self.client._encoding))
        d.addCallback(self.assertEqual, ([fromResponse], [toResponse]))
        return d

    def test_renameFromToEscapesPaths(self):
        """
        L{ftp.FTPClient.rename} issues I{RNTO} and I{RNFR} commands with paths
        escaped according to U{http://cr.yp.to/ftp/filesystem.html}.
        """
        self._testLogin()
        fromFile = '/foo/ba\nr/baz'
        toFile = '/qu\nux'
        self.client.rename(fromFile, toFile)
        self.client.lineReceived(b'350 ')
        self.client.lineReceived(b'250 ')
        self.assertEqual(self.transport.value(), b'RNFR /foo/ba\x00r/baz\r\nRNTO /qu\x00ux\r\n')

    def test_renameFromToFailingOnFirstError(self):
        """
        The L{Deferred} returned by L{ftp.FTPClient.rename} is errbacked with
        L{CommandFailed} if the I{RNFR} command receives an error response code
        (for example, because the file does not exist).
        """
        self._testLogin()
        d = self.client.rename('/spam', '/ham')
        self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'550 Requested file unavailable.\r\n')
        self.assertEqual(self.transport.value(), b'')
        return self.assertFailure(d, ftp.CommandFailed)

    def test_renameFromToFailingOnRenameTo(self):
        """
        The L{Deferred} returned by L{ftp.FTPClient.rename} is errbacked with
        L{CommandFailed} if the I{RNTO} command receives an error response code
        (for example, because the destination directory does not exist).
        """
        self._testLogin()
        d = self.client.rename('/spam', '/ham')
        self.assertEqual(self.transport.value(), b'RNFR /spam\r\n')
        self.transport.clear()
        self.client.lineReceived(b'350 Requested file action pending further information.\r\n')
        self.assertEqual(self.transport.value(), b'RNTO /ham\r\n')
        self.client.lineReceived(b'550 Requested file unavailable.\r\n')
        return self.assertFailure(d, ftp.CommandFailed)

    def test_makeDirectory(self):
        """
        L{ftp.FTPClient.makeDirectory} issues a I{MKD} command and returns a
        L{Deferred} which is called back with the server's response if the
        directory is created.
        """
        self._testLogin()
        d = self.client.makeDirectory('/spam')
        self.assertEqual(self.transport.value(), b'MKD /spam\r\n')
        self.client.lineReceived(b'257 "/spam" created.')
        return d.addCallback(self.assertEqual, ['257 "/spam" created.'])

    def test_makeDirectoryPathEscape(self):
        """
        L{ftp.FTPClient.makeDirectory} escapes the path name it sends according
        to U{http://cr.yp.to/ftp/filesystem.html}.
        """
        self._testLogin()
        d = self.client.makeDirectory('/sp\nam')
        self.assertEqual(self.transport.value(), b'MKD /sp\x00am\r\n')
        self.client.lineReceived(b'257 win')
        return d

    def test_failedMakeDirectory(self):
        """
        L{ftp.FTPClient.makeDirectory} returns a L{Deferred} which is errbacked
        with L{CommandFailed} if the server returns an error response code.
        """
        self._testLogin()
        d = self.client.makeDirectory('/spam')
        self.assertEqual(self.transport.value(), b'MKD /spam\r\n')
        self.client.lineReceived(b'550 PERMISSION DENIED')
        return self.assertFailure(d, ftp.CommandFailed)

    def test_getDirectory(self):
        """
        Test the getDirectory method.

        L{ftp.FTPClient.getDirectory} should return a Deferred which fires with
        the current directory on the server. It wraps PWD command.
        """

        def cbGet(res):
            self.assertEqual(res, '/bar/baz')
        self._testLogin()
        d = self.client.getDirectory().addCallback(cbGet)
        self.assertEqual(self.transport.value(), b'PWD\r\n')
        self.client.lineReceived(b'257 "/bar/baz"')
        return d

    def test_failedGetDirectory(self):
        """
        Test a failure in getDirectory method.

        The behaviour should be the same as PWD.
        """
        self._testLogin()
        d = self.client.getDirectory()
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PWD\r\n')
        self.client.lineReceived(b'550 /bar/baz: No such file or directory')
        return d

    def test_anotherFailedGetDirectory(self):
        """
        Test a different failure in getDirectory method.

        The response should be quoted to be parsed, so it returns an error
        otherwise.
        """
        self._testLogin()
        d = self.client.getDirectory()
        self.assertFailure(d, ftp.CommandFailed)
        self.assertEqual(self.transport.value(), b'PWD\r\n')
        self.client.lineReceived(b'257 /bar/baz')
        return d

    def test_removeFile(self):
        """
        L{ftp.FTPClient.removeFile} sends a I{DELE} command to the server for
        the indicated file and returns a Deferred which fires after the server
        sends a 250 response code.
        """
        self._testLogin()
        d = self.client.removeFile('/tmp/test')
        self.assertEqual(self.transport.value(), b'DELE /tmp/test\r\n')
        response = '250 Requested file action okay, completed.'
        self.client.lineReceived(response.encode(self.client._encoding))
        return d.addCallback(self.assertEqual, [response])

    def test_failedRemoveFile(self):
        """
        If the server returns a response code other than 250 in response to a
        I{DELE} sent by L{ftp.FTPClient.removeFile}, the L{Deferred} returned
        by C{removeFile} is errbacked with a L{Failure} wrapping a
        L{CommandFailed}.
        """
        self._testLogin()
        d = self.client.removeFile('/tmp/test')
        self.assertEqual(self.transport.value(), b'DELE /tmp/test\r\n')
        response = '501 Syntax error in parameters or arguments.'
        self.client.lineReceived(response.encode(self.client._encoding))
        d = self.assertFailure(d, ftp.CommandFailed)
        d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
        return d

    def test_unparsableRemoveFileResponse(self):
        """
        If the server returns a response line which cannot be parsed, the
        L{Deferred} returned by L{ftp.FTPClient.removeFile} is errbacked with a
        L{BadResponse} containing the response.
        """
        self._testLogin()
        d = self.client.removeFile('/tmp/test')
        response = '765 blah blah blah'
        self.client.lineReceived(response.encode(self.client._encoding))
        d = self.assertFailure(d, ftp.BadResponse)
        d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
        return d

    def test_multilineRemoveFileResponse(self):
        """
        If the server returns multiple response lines, the L{Deferred} returned
        by L{ftp.FTPClient.removeFile} is still fired with a true value if the
        ultimate response code is 250.
        """
        self._testLogin()
        d = self.client.removeFile('/tmp/test')
        self.client.lineReceived(b'250-perhaps a progress report')
        self.client.lineReceived(b'250 okay')
        return d.addCallback(self.assertTrue)

    def test_removeDirectory(self):
        """
        L{ftp.FTPClient.removeDirectory} sends a I{RMD} command to the server
        for the indicated directory and returns a Deferred which fires after
        the server sends a 250 response code.
        """
        self._testLogin()
        d = self.client.removeDirectory('/tmp/test')
        self.assertEqual(self.transport.value(), b'RMD /tmp/test\r\n')
        response = '250 Requested file action okay, completed.'
        self.client.lineReceived(response.encode(self.client._encoding))
        return d.addCallback(self.assertEqual, [response])

    def test_failedRemoveDirectory(self):
        """
        If the server returns a response code other than 250 in response to a
        I{RMD} sent by L{ftp.FTPClient.removeDirectory}, the L{Deferred}
        returned by C{removeDirectory} is errbacked with a L{Failure} wrapping
        a L{CommandFailed}.
        """
        self._testLogin()
        d = self.client.removeDirectory('/tmp/test')
        self.assertEqual(self.transport.value(), b'RMD /tmp/test\r\n')
        response = '501 Syntax error in parameters or arguments.'
        self.client.lineReceived(response.encode(self.client._encoding))
        d = self.assertFailure(d, ftp.CommandFailed)
        d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
        return d

    def test_unparsableRemoveDirectoryResponse(self):
        """
        If the server returns a response line which cannot be parsed, the
        L{Deferred} returned by L{ftp.FTPClient.removeDirectory} is errbacked
        with a L{BadResponse} containing the response.
        """
        self._testLogin()
        d = self.client.removeDirectory('/tmp/test')
        response = '765 blah blah blah'
        self.client.lineReceived(response.encode(self.client._encoding))
        d = self.assertFailure(d, ftp.BadResponse)
        d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
        return d

    def test_multilineRemoveDirectoryResponse(self):
        """
        If the server returns multiple response lines, the L{Deferred} returned
        by L{ftp.FTPClient.removeDirectory} is still fired with a true value
         if the ultimate response code is 250.
        """
        self._testLogin()
        d = self.client.removeDirectory('/tmp/test')
        self.client.lineReceived(b'250-perhaps a progress report')
        self.client.lineReceived(b'250 okay')
        return d.addCallback(self.assertTrue)