import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def _testCheckKey(self, filename):
    self.sshDir.child(filename).setContent(self.content)
    user = UsernamePassword(b'user', b'password')
    user.blob = b'foobar'
    self.assertTrue(self.checker.checkKey(user))
    user.blob = b'eggspam'
    self.assertTrue(self.checker.checkKey(user))
    user.blob = b'notallowed'
    self.assertFalse(self.checker.checkKey(user))