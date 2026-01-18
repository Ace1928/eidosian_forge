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
class NewConnectionHelperTests(TestCase):
    """
    Tests for L{_NewConnectionHelper}.
    """

    def test_interface(self):
        """
        L{_NewConnectionHelper} implements L{_ISSHConnectionCreator}.
        """
        self.assertTrue(verifyClass(_ISSHConnectionCreator, _NewConnectionHelper))

    def test_defaultPath(self):
        """
        The default I{known_hosts} path is I{~/.ssh/known_hosts}.
        """
        self.assertEqual('~/.ssh/known_hosts', _NewConnectionHelper._KNOWN_HOSTS)

    def test_defaultKnownHosts(self):
        """
        L{_NewConnectionHelper._knownHosts} is used to create a
        L{KnownHostsFile} if one is not passed to the initializer.
        """
        result = object()
        self.patch(_NewConnectionHelper, '_knownHosts', lambda cls: result)
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None)
        self.assertIs(result, helper.knownHosts)

    def test_readExisting(self):
        """
        Existing entries in the I{known_hosts} file are reflected by the
        L{KnownHostsFile} created by L{_NewConnectionHelper} when none is
        supplied to it.
        """
        key = CommandFactory().publicKeys[b'ssh-rsa']
        path = FilePath(self.mktemp())
        knownHosts = KnownHostsFile(path)
        knownHosts.addHostKey(b'127.0.0.1', key)
        knownHosts.save()
        msg(f'Created known_hosts file at {path.path!r}')
        home = os.path.expanduser('~/')
        default = path.path.replace(home, '~/')
        self.patch(_NewConnectionHelper, '_KNOWN_HOSTS', default)
        msg(f'Patched _KNOWN_HOSTS with {default!r}')
        loaded = _NewConnectionHelper._knownHosts()
        self.assertTrue(loaded.hasHostKey(b'127.0.0.1', key))

    def test_defaultConsoleUI(self):
        """
        If L{None} is passed for the C{ui} parameter to
        L{_NewConnectionHelper}, a L{ConsoleUI} is used.
        """
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None)
        self.assertIsInstance(helper.ui, ConsoleUI)

    def test_ttyConsoleUI(self):
        """
        If L{None} is passed for the C{ui} parameter to L{_NewConnectionHelper}
        and /dev/tty is available, the L{ConsoleUI} used is associated with
        /dev/tty.
        """
        tty = _PTYPath(b'yes')
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None, tty)
        result = self.successResultOf(helper.ui.prompt(b'does this work?'))
        self.assertTrue(result)

    def test_nottyUI(self):
        """
        If L{None} is passed for the C{ui} parameter to L{_NewConnectionHelper}
        and /dev/tty is not available, the L{ConsoleUI} used is associated with
        some file which always produces a C{b"no"} response.
        """
        tty = FilePath(self.mktemp())
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None, tty)
        result = self.successResultOf(helper.ui.prompt(b'did this break?'))
        self.assertFalse(result)

    def test_defaultTTYFilename(self):
        """
        If not passed the name of a tty in the filesystem,
        L{_NewConnectionHelper} uses C{b"/dev/tty"}.
        """
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None)
        self.assertEqual(FilePath(b'/dev/tty'), helper.tty)

    def test_cleanupConnectionNotImmediately(self):
        """
        L{_NewConnectionHelper.cleanupConnection} closes the transport cleanly
        if called with C{immediate} set to C{False}.
        """
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None)
        connection = SSHConnection()
        connection.transport = StringTransport()
        helper.cleanupConnection(connection, False)
        self.assertTrue(connection.transport.disconnecting)

    def test_cleanupConnectionImmediately(self):
        """
        L{_NewConnectionHelper.cleanupConnection} closes the transport with
        C{abortConnection} if called with C{immediate} set to C{True}.
        """

        class Abortable:
            aborted = False

            def abortConnection(self):
                """
                Abort the connection.
                """
                self.aborted = True
        helper = _NewConnectionHelper(None, None, None, None, None, None, None, None, None, None)
        connection = SSHConnection()
        connection.transport = SSHClientTransport()
        connection.transport.transport = Abortable()
        helper.cleanupConnection(connection, True)
        self.assertTrue(connection.transport.transport.aborted)