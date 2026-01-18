import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
class UserDatabaseTests(TestCase, UserDatabaseTestsMixin):
    """
    Tests for L{UserDatabase}.
    """

    def setUp(self):
        """
        Create a L{UserDatabase} with no user data in it.
        """
        self.database = UserDatabase()
        self._counter = SYSTEM_UID_MAX + 1

    def getExistingUserInfo(self):
        """
        Add a new user to C{self.database} and return its information.
        """
        self._counter += 1
        suffix = '_' + str(self._counter)
        username = 'username' + suffix
        password = 'password' + suffix
        uid = self._counter
        gid = self._counter + 1000
        gecos = 'gecos' + suffix
        dir = 'dir' + suffix
        shell = 'shell' + suffix
        self.database.addUser(username, password, uid, gid, gecos, dir, shell)
        return (username, password, uid, gid, gecos, dir, shell)

    def test_addUser(self):
        """
        L{UserDatabase.addUser} accepts seven arguments, one for each field of
        a L{pwd.struct_passwd}, and makes the new record available via
        L{UserDatabase.getpwuid}, L{UserDatabase.getpwnam}, and
        L{UserDatabase.getpwall}.
        """
        username = 'alice'
        password = 'secr3t'
        uid = 123
        gid = 456
        gecos = 'Alice,,,'
        home = '/users/alice'
        shell = '/usr/bin/foosh'
        db = self.database
        db.addUser(username, password, uid, gid, gecos, home, shell)
        for [entry] in [[db.getpwuid(uid)], [db.getpwnam(username)], db.getpwall()]:
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, home)
            self.assertEqual(entry.pw_shell, shell)