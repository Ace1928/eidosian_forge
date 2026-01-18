import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(doImportSkip, importSkipReason)
class SendmsgTests(TestCase):
    """
    Tests for the Python2/3 compatible L{sendmsg} interface.
    """

    def setUp(self):
        """
        Create a pair of UNIX sockets.
        """
        self.input, self.output = socketpair(AF_UNIX)

    def tearDown(self):
        """
        Close the sockets opened by setUp.
        """
        self.input.close()
        self.output.close()

    def test_syscallError(self):
        """
        If the underlying C{sendmsg} call fails, L{send1msg} raises
        L{socket.error} with its errno set to the underlying errno value.
        """
        self.input.close()
        exc = self.assertRaises(error, sendmsg, self.input, b'hello, world')
        self.assertEqual(exc.args[0], errno.EBADF)

    def test_syscallErrorWithControlMessage(self):
        """
        The behavior when the underlying C{sendmsg} call fails is the same
        whether L{sendmsg} is passed ancillary data or not.
        """
        self.input.close()
        exc = self.assertRaises(error, sendmsg, self.input, b'hello, world', [(0, 0, b'0123')], 0)
        self.assertEqual(exc.args[0], errno.EBADF)

    def test_roundtrip(self):
        """
        L{recvmsg} will retrieve a message sent via L{sendmsg}.
        """
        message = b'hello, world!'
        self.assertEqual(len(message), sendmsg(self.input, message))
        result = recvmsg(self.output)
        self.assertEqual(result.data, b'hello, world!')
        self.assertEqual(result.flags, 0)
        self.assertEqual(result.ancillary, [])

    def test_shortsend(self):
        """
        L{sendmsg} returns the number of bytes which it was able to send.
        """
        message = b'x' * 1024 * 1024 * 16
        self.input.setblocking(False)
        sent = sendmsg(self.input, message)
        self.assertTrue(sent < len(message))
        received = recvmsg(self.output, len(message))
        self.assertEqual(len(received[0]), sent)

    def test_roundtripEmptyAncillary(self):
        """
        L{sendmsg} treats an empty ancillary data list the same way it treats
        receiving no argument for the ancillary parameter at all.
        """
        sendmsg(self.input, b'hello, world!', [], 0)
        result = recvmsg(self.output)
        self.assertEqual(result, (b'hello, world!', [], 0))

    @skipIf(dontWaitSkip, 'MSG_DONTWAIT is only known to work as intended on Linux')
    def test_flags(self):
        """
        The C{flags} argument to L{sendmsg} is passed on to the underlying
        C{sendmsg} call, to affect it in whatever way is defined by those
        flags.
        """
        for i in range(8 * 1024):
            try:
                sendmsg(self.input, b'x' * 1024, flags=MSG_DONTWAIT)
            except OSError as e:
                self.assertEqual(e.args[0], errno.EAGAIN)
                break
        else:
            self.fail('Failed to fill up the send buffer, or maybe send1msg blocked for a while')

    @inlineCallbacks
    def test_sendSubProcessFD(self):
        """
        Calling L{sendmsg} with SOL_SOCKET, SCM_RIGHTS, and a platform-endian
        packed file descriptor number should send that file descriptor to a
        different process, where it can be retrieved by using L{recv1msg}.
        """
        sspp = _spawn('pullpipe', self.output.fileno())
        yield sspp.started
        pipeOut, pipeIn = _makePipe()
        self.addCleanup(pipeOut.close)
        self.addCleanup(pipeIn.close)
        with pipeIn:
            sendmsg(self.input, b'blonk', [(SOL_SOCKET, SCM_RIGHTS, pack('i', pipeIn.fileno()))])
        yield sspp.stopped
        self.assertEqual(read(pipeOut.fileno(), 1024), b'Test fixture data: blonk.\n')
        self.assertEqual(read(pipeOut.fileno(), 1024), b'')