import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
class AnotherPOP3Tests(unittest.TestCase):
    """
    Additional L{pop3.POP3} tests.
    """

    def runTest(self, lines, expectedOutput, protocolInstance=None):
        """
        Assert that when C{lines} are delivered to L{pop3.POP3} it responds
        with C{expectedOutput}.

        @param lines: A sequence of L{bytes} representing lines to deliver to
            the server.

        @param expectedOutput: A sequence of L{bytes} representing the
            expected response from the server.

        @param protocolInstance: Instance of L{twisted.mail.pop3.POP3} or
            L{None}. If L{None}, a new DummyPOP3 will be used.

        @return: A L{Deferred} that fires when the lines have been delivered
            and the output checked.
        """
        dummy = protocolInstance if protocolInstance else DummyPOP3()
        client = LineSendingProtocol(lines)
        d = loopback.loopbackAsync(dummy, client)
        return d.addCallback(self._cbRunTest, client, dummy, expectedOutput)

    def _cbRunTest(self, ignored, client, dummy, expectedOutput):
        self.assertEqual(b'\r\n'.join(expectedOutput), b'\r\n'.join(client.response))
        dummy.connectionLost(failure.Failure(Exception('Test harness disconnect')))
        return ignored

    def test_buffer(self):
        """
        Test a lot of different POP3 commands in an extremely pipelined
        scenario.

        This test may cover legitimate behavior, but the intent and
        granularity are not very good.  It would likely be an improvement to
        split it into a number of smaller, more focused tests.
        """
        return self.runTest([b'APOP moshez dummy', b'LIST', b'UIDL', b'RETR 1', b'RETR 2', b'DELE 1', b'RETR 1', b'QUIT'], [b'+OK <moshez>', b'+OK Authentication succeeded', b'+OK 1', b'1 44', b'.', b'+OK ', b'1 0', b'.', b'+OK 44', b'From: moshe', b'To: moshe', b'', b'How are you, friend?', b'.', b'-ERR Bad message number argument', b'+OK ', b'-ERR message deleted', b'+OK '])

    def test_noop(self):
        """
        Test the no-op command.
        """
        return self.runTest([b'APOP spiv dummy', b'NOOP', b'QUIT'], [b'+OK <moshez>', b'+OK Authentication succeeded', b'+OK ', b'+OK '])

    def test_badUTF8CharactersInCommand(self):
        """
        Sending a command with invalid UTF-8 characters
        will raise a L{pop3.POP3Error}.
        """
        error = b'not authenticated yet: cannot do \x81PASS'
        d = self.runTest([b'\x81PASS', b'QUIT'], [b'+OK <moshez>', b'-ERR bad protocol or server: POP3Error: ' + error, b'+OK '])
        errors = self.flushLoggedErrors(pop3.POP3Error)
        self.assertEqual(len(errors), 1)
        return d

    def test_authListing(self):
        """
        L{pop3.POP3} responds to an I{AUTH} command with a list of supported
        authentication types based on its factory's C{challengers}.
        """
        p = DummyPOP3()
        p.factory = internet.protocol.Factory()
        p.factory.challengers = {b'Auth1': None, b'secondAuth': None, b'authLast': None}
        client = LineSendingProtocol([b'AUTH', b'QUIT'])
        d = loopback.loopbackAsync(p, client)
        return d.addCallback(self._cbTestAuthListing, client)

    def _cbTestAuthListing(self, ignored, client):
        self.assertTrue(client.response[1].startswith(b'+OK'))
        self.assertEqual(sorted(client.response[2:5]), [b'AUTH1', b'AUTHLAST', b'SECONDAUTH'])
        self.assertEqual(client.response[5], b'.')

    def run_PASS(self, real_user, real_password, tried_user=None, tried_password=None, after_auth_input=[], after_auth_output=[]):
        """
        Test a login with PASS.

        If L{real_user} matches L{tried_user} and L{real_password} matches
        L{tried_password}, a successful login will be expected.
        Otherwise an unsuccessful login will be expected.

        @type real_user: L{bytes}
        @param real_user: The user to test.

        @type real_password: L{bytes}
        @param real_password: The password of the test user.

        @type tried_user: L{bytes} or L{None}
        @param tried_user: The user to call USER with.
            If None, real_user will be used.

        @type tried_password: L{bytes} or L{None}
        @param tried_password: The password to call PASS with.
            If None, real_password will be used.

        @type after_auth_input: L{list} of l{bytes}
        @param after_auth_input: Extra protocol input after authentication.

        @type after_auth_output: L{list} of l{bytes}
        @param after_auth_output: Extra protocol output after authentication.
        """
        if not tried_user:
            tried_user = real_user
        if not tried_password:
            tried_password = real_password
        response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'-ERR Authentication failed']
        if real_user == tried_user and real_password == tried_password:
            response = [b'+OK <moshez>', b'+OK USER accepted, send PASS', b'+OK Authentication succeeded']
        fullInput = [b' '.join([b'USER', tried_user]), b' '.join([b'PASS', tried_password])]
        fullInput += after_auth_input + [b'QUIT']
        response += after_auth_output + [b'+OK ']
        return self.runTest(fullInput, response, protocolInstance=DummyPOP3Auth(real_user, real_password))

    def run_PASS_before_USER(self, password):
        """
        Test protocol violation produced by calling PASS before USER.
        @type password: L{bytes}
        @param password: A password to test.
        """
        return self.runTest([b' '.join([b'PASS', password]), b'QUIT'], [b'+OK <moshez>', b'-ERR USER required before PASS', b'+OK '])

    def test_illegal_PASS_before_USER(self):
        """
        Test PASS before USER with a wrong password.
        """
        return self.run_PASS_before_USER(b'fooz')

    def test_empty_PASS_before_USER(self):
        """
        Test PASS before USER with an empty password.
        """
        return self.run_PASS_before_USER(b'')

    def test_one_space_PASS_before_USER(self):
        """
        Test PASS before USER with an password that is a space.
        """
        return self.run_PASS_before_USER(b' ')

    def test_space_PASS_before_USER(self):
        """
        Test PASS before USER with a password containing a space.
        """
        return self.run_PASS_before_USER(b'fooz barz')

    def test_multiple_spaces_PASS_before_USER(self):
        """
        Test PASS before USER with a password containing multiple spaces.
        """
        return self.run_PASS_before_USER(b'fooz barz asdf')

    def test_other_whitespace_PASS_before_USER(self):
        """
        Test PASS before USER with a password containing tabs and spaces.
        """
        return self.run_PASS_before_USER(b'fooz barz\tcrazy@! \t ')

    def test_good_PASS(self):
        """
        Test PASS with a good password.
        """
        return self.run_PASS(b'testuser', b'fooz')

    def test_space_PASS(self):
        """
        Test PASS with a password containing a space.
        """
        return self.run_PASS(b'testuser', b'fooz barz')

    def test_multiple_spaces_PASS(self):
        """
        Test PASS with a password containing a space.
        """
        return self.run_PASS(b'testuser', b'fooz barz asdf')

    def test_other_whitespace_PASS(self):
        """
        Test PASS with a password containing tabs and spaces.
        """
        return self.run_PASS(b'testuser', b'fooz barz\tcrazy@! \t ')

    def test_pass_wrong_user(self):
        """
        Test PASS with a wrong user.
        """
        return self.run_PASS(b'testuser', b'fooz', tried_user=b'wronguser')

    def test_wrong_PASS(self):
        """
        Test PASS with a wrong password.
        """
        return self.run_PASS(b'testuser', b'fooz', tried_password=b'barz')

    def test_wrong_space_PASS(self):
        """
        Test PASS with a password containing a space.
        """
        return self.run_PASS(b'testuser', b'fooz barz', tried_password=b'foozbarz ')

    def test_wrong_multiple_spaces_PASS(self):
        """
        Test PASS with a password containing a space.
        """
        return self.run_PASS(b'testuser', b'fooz barz asdf', tried_password=b'foozbarz   ')

    def test_wrong_other_whitespace_PASS(self):
        """
        Test PASS with a password containing tabs and spaces.
        """
        return self.run_PASS(b'testuser', b'fooz barz\tcrazy@! \t ')

    def test_wrong_command(self):
        """
        After logging in, test a dummy command that is not defined.
        """
        extra_input = [b'DUMMY COMMAND']
        extra_output = [b' '.join([b'-ERR bad protocol or server: POP3Error:', b'Unknown protocol command: DUMMY'])]
        return self.run_PASS(b'testuser', b'testpassword', after_auth_input=extra_input, after_auth_output=extra_output).addCallback(self.flushLoggedErrors, pop3.POP3Error)