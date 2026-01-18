import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
@skipIf(not crypt, 'Required module is unavailable: crypt')
class CryptTests(TestCase):
    """
    L{crypt} has functions for encrypting password.
    """

    def test_verifyCryptedPassword(self):
        """
        L{cred_unix.verifyCryptedPassword}
        """
        password = 'sample password ^%$'
        for salt in (None, 'ab'):
            try:
                cryptedCorrect = crypt.crypt(password, salt)
                if isinstance(cryptedCorrect, bytes):
                    cryptedCorrect = cryptedCorrect.decode('utf-8')
            except TypeError:
                continue
            cryptedIncorrect = '$1x1234'
            self.assertTrue(cred_unix.verifyCryptedPassword(cryptedCorrect, password))
            self.assertFalse(cred_unix.verifyCryptedPassword(cryptedIncorrect, password))
        for method in ('METHOD_SHA512', 'METHOD_SHA256', 'METHOD_MD5', 'METHOD_CRYPT'):
            cryptMethod = getattr(crypt, method, None)
            if not cryptMethod:
                continue
            password = 'interesting password xyz'
            crypted = crypt.crypt(password, cryptMethod)
            if isinstance(crypted, bytes):
                crypted = crypted.decode('utf-8')
            incorrectCrypted = crypted + 'blahfooincorrect'
            result = cred_unix.verifyCryptedPassword(crypted, password)
            self.assertTrue(result)
            result = cred_unix.verifyCryptedPassword(crypted.encode('utf-8'), password.encode('utf-8'))
            self.assertTrue(result)
            result = cred_unix.verifyCryptedPassword(incorrectCrypted, password)
            self.assertFalse(result)
            result = cred_unix.verifyCryptedPassword(incorrectCrypted.encode('utf-8'), password.encode('utf-8'))
            self.assertFalse(result)

    def test_verifyCryptedPasswordOSError(self):
        """
        L{cred_unix.verifyCryptedPassword} when OSError is raised
        """

        def mockCrypt(password, salt):
            raise OSError('')
        password = 'sample password ^%$'
        cryptedCorrect = crypt.crypt(password, 'ab')
        self.patch(crypt, 'crypt', mockCrypt)
        self.assertFalse(cred_unix.verifyCryptedPassword(cryptedCorrect, password))