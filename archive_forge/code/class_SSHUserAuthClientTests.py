from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class SSHUserAuthClientTests(unittest.TestCase):
    """
    Tests for SSHUserAuthClient.
    """
    if keys is None:
        skip = 'cannot run without cryptography'

    def setUp(self):
        self.authClient = ClientUserAuth(b'foo', FakeTransport.Service())
        self.authClient.transport = FakeTransport(None)
        self.authClient.transport.sessionID = b'test'
        self.authClient.serviceStarted()

    def tearDown(self):
        self.authClient.serviceStopped()
        self.authClient = None

    def test_init(self):
        """
        Test that client is initialized properly.
        """
        self.assertEqual(self.authClient.user, b'foo')
        self.assertEqual(self.authClient.instance.name, b'nancy')
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_USERAUTH_SUCCESS(self):
        """
        Test that the client succeeds properly.
        """
        instance = [None]

        def stubSetService(service):
            instance[0] = service
        self.authClient.transport.setService = stubSetService
        self.authClient.ssh_USERAUTH_SUCCESS(b'')
        self.assertEqual(instance[0], self.authClient.instance)

    def test_publickey(self):
        """
        Test that the client can authenticate with a public key.
        """
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-dss') + NS(keys.Key.fromString(keydata.publicDSA_openssh).blob())))
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
        blob = NS(keys.Key.fromString(keydata.publicRSA_openssh).blob())
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + blob))
        self.authClient.ssh_USERAUTH_PK_OK(NS(b'ssh-rsa') + NS(keys.Key.fromString(keydata.publicRSA_openssh).blob()))
        sigData = NS(self.authClient.transport.sessionID) + bytes((userauth.MSG_USERAUTH_REQUEST,)) + NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob
        obj = keys.Key.fromString(keydata.privateRSA_openssh)
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob + NS(obj.sign(sigData))))

    def test_publickey_without_privatekey(self):
        """
        If the SSHUserAuthClient doesn't return anything from signData,
        the client should start the authentication over again by requesting
        'none' authentication.
        """
        authClient = ClientAuthWithoutPrivateKey(b'foo', FakeTransport.Service())
        authClient.transport = FakeTransport(None)
        authClient.transport.sessionID = b'test'
        authClient.serviceStarted()
        authClient.tryAuth(b'publickey')
        authClient.transport.packets = []
        self.assertIsNone(authClient.ssh_USERAUTH_PK_OK(b''))
        self.assertEqual(authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_no_publickey(self):
        """
        If there's no public key, auth_publickey should return a Deferred
        called back with a False value.
        """
        self.authClient.getPublicKey = lambda x: None
        d = self.authClient.tryAuth(b'publickey')

        def check(result):
            self.assertFalse(result)
        return d.addCallback(check)

    def test_password(self):
        """
        Test that the client can authentication with a password.  This
        includes changing the password.
        """
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
        self.authClient.ssh_USERAUTH_PK_OK(NS(b'') + NS(b''))
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\xff' + NS(b'foo') * 2))

    def test_no_password(self):
        """
        If getPassword returns None, tryAuth should return False.
        """
        self.authClient.getPassword = lambda: None
        self.assertFalse(self.authClient.tryAuth(b'password'))

    def test_keyboardInteractive(self):
        """
        Make sure that the client can authenticate with the keyboard
        interactive method.
        """
        self.authClient.ssh_USERAUTH_PK_OK_keyboard_interactive(NS(b'') + NS(b'') + NS(b'') + b'\x00\x00\x00\x01' + NS(b'Password: ') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_INFO_RESPONSE, b'\x00\x00\x00\x02' + NS(b'foo') + NS(b'foo')))

    def test_USERAUTH_PK_OK_unknown_method(self):
        """
        If C{SSHUserAuthClient} gets a MSG_USERAUTH_PK_OK packet when it's not
        expecting it, it should fail the current authentication and move on to
        the next type.
        """
        self.authClient.lastAuth = b'unknown'
        self.authClient.transport.packets = []
        self.authClient.ssh_USERAUTH_PK_OK(b'')
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_USERAUTH_FAILURE_sorting(self):
        """
        ssh_USERAUTH_FAILURE should sort the methods by their position
        in SSHUserAuthClient.preferredOrder.  Methods that are not in
        preferredOrder should be sorted at the end of that list.
        """

        def auth_firstmethod():
            self.authClient.transport.sendPacket(255, b'here is data')

        def auth_anothermethod():
            self.authClient.transport.sendPacket(254, b'other data')
            return True
        self.authClient.auth_firstmethod = auth_firstmethod
        self.authClient.auth_anothermethod = auth_anothermethod
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'anothermethod,password') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'firstmethod,anothermethod,password') + b'\xff')
        self.assertEqual(self.authClient.transport.packets[-2:], [(255, b'here is data'), (254, b'other data')])

    def test_disconnectIfNoMoreAuthentication(self):
        """
        If there are no more available user authentication messages,
        the SSHUserAuthClient should disconnect with code
        DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE.
        """
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\xff')
        self.assertEqual(self.authClient.transport.packets[-1], (transport.MSG_DISCONNECT, b'\x00\x00\x00\x0e' + NS(b'no more authentication methods available') + b'\x00\x00\x00\x00'))

    def test_ebAuth(self):
        """
        _ebAuth (the generic authentication error handler) should send
        a request for the 'none' authentication method.
        """
        self.authClient.transport.packets = []
        self.authClient._ebAuth(None)
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_defaults(self):
        """
        getPublicKey() should return None.  getPrivateKey() should return a
        failed Deferred.  getPassword() should return a failed Deferred.
        getGenericAnswers() should return a failed Deferred.
        """
        authClient = userauth.SSHUserAuthClient(b'foo', FakeTransport.Service())
        self.assertIsNone(authClient.getPublicKey())

        def check(result):
            result.trap(NotImplementedError)
            d = authClient.getPassword()
            return d.addCallback(self.fail).addErrback(check2)

        def check2(result):
            result.trap(NotImplementedError)
            d = authClient.getGenericAnswers(None, None, None)
            return d.addCallback(self.fail).addErrback(check3)

        def check3(result):
            result.trap(NotImplementedError)
        d = authClient.getPrivateKey()
        return d.addCallback(self.fail).addErrback(check)