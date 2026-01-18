from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class CredTests(unittest.TestCase):
    """
    Tests for the meat of L{twisted.cred} -- realms, portals, avatars, and
    checkers.
    """

    def setUp(self):
        self.realm = TestRealm()
        self.portal = portal.Portal(self.realm)
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.checker.addUser(b'bob', b'hello')
        self.portal.registerChecker(self.checker)

    def test_listCheckers(self):
        """
        The checkers in a portal can check only certain types of credentials.
        Since this portal has
        L{checkers.InMemoryUsernamePasswordDatabaseDontUse} registered, it
        """
        expected = [credentials.IUsernamePassword, credentials.IUsernameHashedPassword]
        got = self.portal.listCredentialsInterfaces()
        self.assertEqual(sorted(got), sorted(expected))

    def test_basicLogin(self):
        """
        Calling C{login} on a portal with correct credentials and an interface
        that the portal's realm supports works.
        """
        login = self.successResultOf(self.portal.login(credentials.UsernamePassword(b'bob', b'hello'), self, ITestable))
        iface, impl, logout = login
        self.assertEqual(iface, ITestable)
        self.assertTrue(iface.providedBy(impl), f'{impl} does not implement {iface}')
        self.assertTrue(impl.original.loggedIn)
        self.assertTrue(not impl.original.loggedOut)
        logout()
        self.assertTrue(impl.original.loggedOut)

    def test_derivedInterface(self):
        """
        Logging in with correct derived credentials and an interface
        that the portal's realm supports works.
        """
        login = self.successResultOf(self.portal.login(DerivedCredentials(b'bob', b'hello'), self, ITestable))
        iface, impl, logout = login
        self.assertEqual(iface, ITestable)
        self.assertTrue(iface.providedBy(impl), f'{impl} does not implement {iface}')
        self.assertTrue(impl.original.loggedIn)
        self.assertTrue(not impl.original.loggedOut)
        logout()
        self.assertTrue(impl.original.loggedOut)

    def test_failedLoginPassword(self):
        """
        Calling C{login} with incorrect credentials (in this case a wrong
        password) causes L{error.UnauthorizedLogin} to be raised.
        """
        login = self.failureResultOf(self.portal.login(credentials.UsernamePassword(b'bob', b'h3llo'), self, ITestable))
        self.assertTrue(login)
        self.assertEqual(error.UnauthorizedLogin, login.type)

    def test_failedLoginName(self):
        """
        Calling C{login} with incorrect credentials (in this case no known
        user) causes L{error.UnauthorizedLogin} to be raised.
        """
        login = self.failureResultOf(self.portal.login(credentials.UsernamePassword(b'jay', b'hello'), self, ITestable))
        self.assertTrue(login)
        self.assertEqual(error.UnauthorizedLogin, login.type)