from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class OnDiskDatabaseTests(unittest.TestCase):
    users = [(b'user1', b'pass1'), (b'user2', b'pass2'), (b'user3', b'pass3')]

    def setUp(self):
        self.dbfile = self.mktemp()
        with open(self.dbfile, 'wb') as f:
            for u, p in self.users:
                f.write(u + b':' + p + b'\n')

    def test_getUserNonexistentDatabase(self):
        """
        A missing db file will cause a permanent rejection of authorization
        attempts.
        """
        self.db = checkers.FilePasswordDB('test_thisbetternoteverexist.db')
        self.assertRaises(error.UnauthorizedLogin, self.db.getUser, 'user')

    def testUserLookup(self):
        self.db = checkers.FilePasswordDB(self.dbfile)
        for u, p in self.users:
            self.assertRaises(KeyError, self.db.getUser, u.upper())
            self.assertEqual(self.db.getUser(u), (u, p))

    def testCaseInSensitivity(self):
        self.db = checkers.FilePasswordDB(self.dbfile, caseSensitive=False)
        for u, p in self.users:
            self.assertEqual(self.db.getUser(u.upper()), (u, p))

    def testRequestAvatarId(self):
        self.db = checkers.FilePasswordDB(self.dbfile)
        creds = [credentials.UsernamePassword(u, p) for u, p in self.users]
        d = defer.gatherResults([defer.maybeDeferred(self.db.requestAvatarId, c) for c in creds])
        d.addCallback(self.assertEqual, [u for u, p in self.users])
        return d

    def testRequestAvatarId_hashed(self):
        self.db = checkers.FilePasswordDB(self.dbfile)
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        creds = [UsernameHashedPassword(u, p) for u, p in self.users]
        d = defer.gatherResults([defer.maybeDeferred(self.db.requestAvatarId, c) for c in creds])
        d.addCallback(self.assertEqual, [u for u, p in self.users])
        return d