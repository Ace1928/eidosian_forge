import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
class ShadowDatabaseTestsMixin:
    """
    L{ShadowDatabaseTestsMixin} defines tests which apply to any shadow user
    database implementation.  Subclasses should mix it in, implement C{setUp} to
    create C{self.database} bound to a shadow user database instance, and
    implement C{getExistingUserInfo} to return information about a user (such
    information should be unique per test method).
    """

    def test_getspnam(self):
        """
        L{getspnam} accepts a username and returns the user record associated
        with it.
        """
        for i in range(2):
            username, password, lastChange, min, max, warn, inact, expire, flag = self.getExistingUserInfo()
            entry = self.database.getspnam(username)
            self.assertEqual(entry.sp_nam, username)
            self.assertEqual(entry.sp_pwd, password)
            self.assertEqual(entry.sp_lstchg, lastChange)
            self.assertEqual(entry.sp_min, min)
            self.assertEqual(entry.sp_max, max)
            self.assertEqual(entry.sp_warn, warn)
            self.assertEqual(entry.sp_inact, inact)
            self.assertEqual(entry.sp_expire, expire)
            self.assertEqual(entry.sp_flag, flag)

    def test_noSuchName(self):
        """
        I{getspnam} raises L{KeyError} when passed a username which does not
        exist in the user database.
        """
        self.assertRaises(KeyError, self.database.getspnam, 'alice')

    def test_getspnamBytes(self):
        """
        I{getspnam} raises L{TypeError} when passed a L{bytes}, just like
        L{spwd.getspnam}.
        """
        self.assertRaises(TypeError, self.database.getspnam, b'i-am-bytes')

    def test_recordLength(self):
        """
        The shadow user record returned by I{getspnam} and I{getspall} has a
        length.
        """
        db = self.database
        username = self.getExistingUserInfo()[0]
        for entry in [db.getspnam(username), db.getspall()[0]]:
            self.assertIsInstance(len(entry), int)
            self.assertEqual(len(entry), 9)

    def test_recordIndexable(self):
        """
        The shadow user record returned by I{getpwnam} and I{getspall} is
        indexable, with successive indexes starting from 0 corresponding to the
        values of the C{sp_nam}, C{sp_pwd}, C{sp_lstchg}, C{sp_min}, C{sp_max},
        C{sp_warn}, C{sp_inact}, C{sp_expire}, and C{sp_flag} attributes,
        respectively.
        """
        db = self.database
        username, password, lastChange, min, max, warn, inact, expire, flag = self.getExistingUserInfo()
        for entry in [db.getspnam(username), db.getspall()[0]]:
            self.assertEqual(entry[0], username)
            self.assertEqual(entry[1], password)
            self.assertEqual(entry[2], lastChange)
            self.assertEqual(entry[3], min)
            self.assertEqual(entry[4], max)
            self.assertEqual(entry[5], warn)
            self.assertEqual(entry[6], inact)
            self.assertEqual(entry[7], expire)
            self.assertEqual(entry[8], flag)
            self.assertEqual(len(entry), len(list(entry)))
            self.assertRaises(IndexError, getitem, entry, 9)