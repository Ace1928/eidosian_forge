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
class ClientUserAuth(userauth.SSHUserAuthClient):
    """
    A mock user auth client.
    """

    def getPublicKey(self):
        """
        If this is the first time we've been called, return a blob for
        the DSA key.  Otherwise, return a blob
        for the RSA key.
        """
        if self.lastPublicKey:
            return keys.Key.fromString(keydata.publicRSA_openssh)
        else:
            return defer.succeed(keys.Key.fromString(keydata.publicDSA_openssh))

    def getPrivateKey(self):
        """
        Return the private key object for the RSA key.
        """
        return defer.succeed(keys.Key.fromString(keydata.privateRSA_openssh))

    def getPassword(self, prompt=None):
        """
        Return 'foo' as the password.
        """
        return defer.succeed(b'foo')

    def getGenericAnswers(self, name, information, answers):
        """
        Return 'foo' as the answer to two questions.
        """
        return defer.succeed(('foo', 'foo'))