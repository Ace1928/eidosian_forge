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
class BasicFTPServerTests(FTPServerTestCase):
    """
    Basic functionality of FTP server.
    """

    def test_tooManyConnections(self):
        """
        When the connection limit is reached, the server should send an
        appropriate response
        """
        self.factory.connectionLimit = 1
        cc = protocol.ClientCreator(reactor, _BufferingProtocol)
        d = cc.connectTCP('127.0.0.1', self.port.getHost().port)

        @d.addCallback
        def gotClient(proto):
            return proto.d

        @d.addCallback
        def onConnectionLost(proto):
            self.assertEqual(b'421 Too many users right now, try again in a few minutes.\r\n', proto.buffer)
        return d

    def test_NotLoggedInReply(self):
        """
        When not logged in, most commands other than USER and PASS should
        get NOT_LOGGED_IN errors, but some can be called before USER and PASS.
        """
        loginRequiredCommandList = ['CDUP', 'CWD', 'LIST', 'MODE', 'PASV', 'PWD', 'RETR', 'STRU', 'SYST', 'TYPE']
        loginNotRequiredCommandList = ['FEAT']

        def checkFailResponse(exception, command):
            failureResponseLines = exception.args[0]
            self.assertTrue(failureResponseLines[-1].startswith('530'), "%s - Response didn't start with 530: %r" % (command, failureResponseLines[-1]))

        def checkPassResponse(result, command):
            result = result[0]
            self.assertFalse(result.startswith('530'), '%s - Response start with 530: %r' % (command, result))
        deferreds = []
        for command in loginRequiredCommandList:
            deferred = self.client.queueStringCommand(command)
            self.assertFailure(deferred, ftp.CommandFailed)
            deferred.addCallback(checkFailResponse, command)
            deferreds.append(deferred)
        for command in loginNotRequiredCommandList:
            deferred = self.client.queueStringCommand(command)
            deferred.addCallback(checkPassResponse, command)
            deferreds.append(deferred)
        return defer.DeferredList(deferreds, fireOnOneErrback=True)

    def test_PASSBeforeUSER(self):
        """
        Issuing PASS before USER should give an error.
        """
        return self.assertCommandFailed('PASS foo', ['503 Incorrect sequence of commands: USER required before PASS'])

    def test_NoParamsForUSER(self):
        """
        Issuing USER without a username is a syntax error.
        """
        return self.assertCommandFailed('USER', ['500 Syntax error: USER requires an argument.'])

    def test_NoParamsForPASS(self):
        """
        Issuing PASS without a password is a syntax error.
        """
        d = self.client.queueStringCommand('USER foo')
        return self.assertCommandFailed('PASS', ['500 Syntax error: PASS requires an argument.'], chainDeferred=d)

    def test_loginError(self):
        """
        Unexpected exceptions from the login handler are caught
        """

        def _fake_loginhandler(*args, **kwargs):
            return defer.fail(AssertionError('test exception'))
        self.serverProtocol.portal.login = _fake_loginhandler
        d = self.client.queueStringCommand('USER foo')
        self.assertCommandFailed('PASS bar', ['550 Requested action not taken: internal server error'], chainDeferred=d)

        @d.addCallback
        def checkLogs(result):
            logs = self.flushLoggedErrors()
            self.assertEqual(1, len(logs))
            self.assertIsInstance(logs[0].value, AssertionError)
        return d

    def test_AnonymousLogin(self):
        """
        Login with userid 'anonymous'
        """
        return self._anonymousLogin()

    def test_Quit(self):
        """
        Issuing QUIT should return a 221 message.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        return self.assertCommandResponse('QUIT', ['221 Goodbye.'], chainDeferred=d)

    def test_AnonymousLoginDenied(self):
        """
        Reconfigure the server to disallow anonymous access, and to have an
        IUsernamePassword checker that always rejects.

        @return: L{Deferred} of command response
        """
        self.factory.allowAnonymous = False
        denyAlwaysChecker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.factory.portal.registerChecker(denyAlwaysChecker, credentials.IUsernamePassword)
        d = self.assertCommandResponse('USER anonymous', ['331 Password required for anonymous.'])
        d = self.assertCommandFailed('PASS test@twistedmatrix.com', ['530 Sorry, Authentication failed.'], chainDeferred=d)
        d = self.assertCommandFailed('PWD', ['530 Please login with USER and PASS.'], chainDeferred=d)
        return d

    def test_anonymousWriteDenied(self):
        """
        When an anonymous user attempts to edit the server-side filesystem, they
        will receive a 550 error with a descriptive message.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        return self.assertCommandFailed('MKD newdir', ['550 Anonymous users are forbidden to change the filesystem'], chainDeferred=d)

    def test_UnknownCommand(self):
        """
        Send an invalid command.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        return self.assertCommandFailed('GIBBERISH', ["502 Command 'GIBBERISH' not implemented"], chainDeferred=d)

    def test_RETRBeforePORT(self):
        """
        Send RETR before sending PORT.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        return self.assertCommandFailed('RETR foo', ['503 Incorrect sequence of commands: PORT or PASV required before RETR'], chainDeferred=d)

    def test_STORBeforePORT(self):
        """
        Send STOR before sending PORT.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        return self.assertCommandFailed('STOR foo', ['503 Incorrect sequence of commands: PORT or PASV required before STOR'], chainDeferred=d)

    def test_BadCommandArgs(self):
        """
        Send command with bad arguments.

        @return: L{Deferred} of command response
        """
        d = self._anonymousLogin()
        self.assertCommandFailed('MODE z', ["504 Not implemented for parameter 'z'."], chainDeferred=d)
        self.assertCommandFailed('STRU I', ["504 Not implemented for parameter 'I'."], chainDeferred=d)
        return d

    def test_DecodeHostPort(self):
        """
        Decode a host port.
        """
        self.assertEqual(ftp.decodeHostPort('25,234,129,22,100,23'), ('25.234.129.22', 25623))
        nums = range(6)
        for i in range(6):
            badValue = list(nums)
            badValue[i] = 256
            s = ','.join(map(str, badValue))
            self.assertRaises(ValueError, ftp.decodeHostPort, s)

    def test_PASV(self):
        """
        When the client sends the command C{PASV}, the server responds with a
        host and port, and is listening on that port.
        """
        d = self._anonymousLogin()
        d.addCallback(lambda _: self.client.queueStringCommand('PASV'))

        def cb(responseLines):
            """
            Extract the host and port from the resonse, and
            verify the server is listening of the port it claims to be.
            """
            host, port = ftp.decodeHostPort(responseLines[-1][4:])
            self.assertEqual(port, self.serverProtocol.dtpPort.getHost().port)
        d.addCallback(cb)
        d.addCallback(lambda _: self.serverProtocol.transport.loseConnection())
        return d

    def test_SYST(self):
        """
        SYST command will always return UNIX Type: L8
        """
        d = self._anonymousLogin()
        self.assertCommandResponse('SYST', ['215 UNIX Type: L8'], chainDeferred=d)
        return d

    def test_RNFRandRNTO(self):
        """
        Sending the RNFR command followed by RNTO, with valid filenames, will
        perform a successful rename operation.
        """
        self.dirPath.child(self.username).createDirectory()
        self.dirPath.child(self.username).child('foo').touch()
        d = self._userLogin()
        self.assertCommandResponse('RNFR foo', ['350 Requested file action pending further information.'], chainDeferred=d)
        self.assertCommandResponse('RNTO bar', ['250 Requested File Action Completed OK'], chainDeferred=d)

        def check_rename(result):
            self.assertTrue(self.dirPath.child(self.username).child('bar').exists())
            return result
        d.addCallback(check_rename)
        return d

    def test_RNFRwithoutRNTO(self):
        """
        Sending the RNFR command followed by any command other than RNTO
        should return an error informing users that RNFR should be followed
        by RNTO.
        """
        d = self._anonymousLogin()
        self.assertCommandResponse('RNFR foo', ['350 Requested file action pending further information.'], chainDeferred=d)
        self.assertCommandFailed('OTHER don-tcare', ['503 Incorrect sequence of commands: RNTO required after RNFR'], chainDeferred=d)
        return d

    def test_portRangeForwardError(self):
        """
        Exceptions other than L{error.CannotListenError} which are raised by
        C{listenFactory} should be raised to the caller of L{FTP.getDTPPort}.
        """

        def listenFactory(portNumber, factory):
            raise RuntimeError()
        self.serverProtocol.listenFactory = listenFactory
        self.assertRaises(RuntimeError, self.serverProtocol.getDTPPort, protocol.Factory())

    def test_portRange(self):
        """
        L{FTP.passivePortRange} should determine the ports which
        L{FTP.getDTPPort} attempts to bind. If no port from that iterator can
        be bound, L{error.CannotListenError} should be raised, otherwise the
        first successful result from L{FTP.listenFactory} should be returned.
        """

        def listenFactory(portNumber, factory):
            if portNumber in (22032, 22033, 22034):
                raise error.CannotListenError('localhost', portNumber, 'error')
            return portNumber
        self.serverProtocol.listenFactory = listenFactory
        port = self.serverProtocol.getDTPPort(protocol.Factory())
        self.assertEqual(port, 0)
        self.serverProtocol.passivePortRange = range(22032, 65536)
        port = self.serverProtocol.getDTPPort(protocol.Factory())
        self.assertEqual(port, 22035)
        self.serverProtocol.passivePortRange = range(22032, 22035)
        self.assertRaises(error.CannotListenError, self.serverProtocol.getDTPPort, protocol.Factory())

    def test_portRangeInheritedFromFactory(self):
        """
        The L{FTP} instances created by L{ftp.FTPFactory.buildProtocol} have
        their C{passivePortRange} attribute set to the same object the
        factory's C{passivePortRange} attribute is set to.
        """
        portRange = range(2017, 2031)
        self.factory.passivePortRange = portRange
        protocol = self.factory.buildProtocol(None)
        self.assertEqual(portRange, protocol.wrappedProtocol.passivePortRange)

    def test_FEAT(self):
        """
        When the server receives 'FEAT', it should report the list of supported
        features. (Additionally, ensure that the server reports various
        particular features that are supported by all Twisted FTP servers.)
        """
        d = self.client.queueStringCommand('FEAT')

        def gotResponse(responseLines):
            self.assertEqual('211-Features:', responseLines[0])
            self.assertIn(' MDTM', responseLines)
            self.assertIn(' PASV', responseLines)
            self.assertIn(' TYPE A;I', responseLines)
            self.assertIn(' SIZE', responseLines)
            self.assertEqual('211 End', responseLines[-1])
        return d.addCallback(gotResponse)

    def test_OPTS(self):
        """
        When the server receives 'OPTS something', it should report
        that the FTP server does not support the option called 'something'.
        """
        d = self._anonymousLogin()
        self.assertCommandFailed('OPTS something', ["502 Option 'something' not implemented."], chainDeferred=d)
        return d

    def test_STORreturnsErrorFromOpen(self):
        """
        Any FTP error raised inside STOR while opening the file is returned
        to the client.
        """
        self.dirPath.child(self.username).createDirectory()
        self.dirPath.child(self.username).child('folder').createDirectory()
        d = self._userLogin()

        def sendPASV(result):
            """
            Send the PASV command required before port.
            """
            return self.client.queueStringCommand('PASV')

        def mockDTPInstance(result):
            """
            Fake an incoming connection and create a mock DTPInstance so
            that PORT command will start processing the request.
            """
            self.serverProtocol.dtpFactory.deferred.callback(None)
            self.serverProtocol.dtpInstance = object()
            return result
        d.addCallback(sendPASV)
        d.addCallback(mockDTPInstance)
        self.assertCommandFailed('STOR folder', ['550 folder: is a directory'], chainDeferred=d)
        return d

    def test_STORunknownErrorBecomesFileNotFound(self):
        """
        Any non FTP error raised inside STOR while opening the file is
        converted into FileNotFound error and returned to the client together
        with the path.

        The unknown error is logged.
        """
        d = self._userLogin()

        def failingOpenForWriting(ignore):
            """
            Override openForWriting.

            @param ignore: ignored, used for callback
            @return: an error
            """
            return defer.fail(AssertionError())

        def sendPASV(result):
            """
            Send the PASV command required before port.

            @param result: parameter used in L{Deferred}
            """
            return self.client.queueStringCommand('PASV')

        def mockDTPInstance(result):
            """
            Fake an incoming connection and create a mock DTPInstance so
            that PORT command will start processing the request.

            @param result: parameter used in L{Deferred}
            """
            self.serverProtocol.dtpFactory.deferred.callback(None)
            self.serverProtocol.dtpInstance = object()
            self.serverProtocol.shell.openForWriting = failingOpenForWriting
            return result

        def checkLogs(result):
            """
            Check that unknown errors are logged.

            @param result: parameter used in L{Deferred}
            """
            logs = self.flushLoggedErrors()
            self.assertEqual(1, len(logs))
            self.assertIsInstance(logs[0].value, AssertionError)
        d.addCallback(sendPASV)
        d.addCallback(mockDTPInstance)
        self.assertCommandFailed('STOR something', ['550 something: No such file or directory.'], chainDeferred=d)
        d.addCallback(checkLogs)
        return d