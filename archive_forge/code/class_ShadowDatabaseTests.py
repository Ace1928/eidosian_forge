import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
class ShadowDatabaseTests(TestCase, ShadowDatabaseTestsMixin):
    """
    Tests for L{ShadowDatabase}.
    """

    def setUp(self):
        """
        Create a L{ShadowDatabase} with no user data in it.
        """
        self.database = ShadowDatabase()
        self._counter = 0

    def getExistingUserInfo(self):
        """
        Add a new user to C{self.database} and return its information.
        """
        self._counter += 1
        suffix = '_' + str(self._counter)
        username = 'username' + suffix
        password = 'password' + suffix
        lastChange = self._counter + 1
        min = self._counter + 2
        max = self._counter + 3
        warn = self._counter + 4
        inact = self._counter + 5
        expire = self._counter + 6
        flag = self._counter + 7
        self.database.addUser(username, password, lastChange, min, max, warn, inact, expire, flag)
        return (username, password, lastChange, min, max, warn, inact, expire, flag)

    def test_addUser(self):
        """
        L{UserDatabase.addUser} accepts seven arguments, one for each field of
        a L{pwd.struct_passwd}, and makes the new record available via
        L{UserDatabase.getpwuid}, L{UserDatabase.getpwnam}, and
        L{UserDatabase.getpwall}.
        """
        username = 'alice'
        password = 'secr3t'
        lastChange = 17
        min = 42
        max = 105
        warn = 12
        inact = 3
        expire = 400
        flag = 3
        db = self.database
        db.addUser(username, password, lastChange, min, max, warn, inact, expire, flag)
        for [entry] in [[db.getspnam(username)], db.getspall()]:
            self.assertEqual(entry.sp_nam, username)
            self.assertEqual(entry.sp_pwd, password)
            self.assertEqual(entry.sp_lstchg, lastChange)
            self.assertEqual(entry.sp_min, min)
            self.assertEqual(entry.sp_max, max)
            self.assertEqual(entry.sp_warn, warn)
            self.assertEqual(entry.sp_inact, inact)
            self.assertEqual(entry.sp_expire, expire)
            self.assertEqual(entry.sp_flag, flag)