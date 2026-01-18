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
class OldClientAuth(userauth.SSHUserAuthClient):
    """
    The old SSHUserAuthClient returned a cryptography key object from
    getPrivateKey() and a string from getPublicKey
    """

    def getPrivateKey(self):
        return defer.succeed(keys.Key.fromString(keydata.privateRSA_openssh).keyObject)

    def getPublicKey(self):
        return keys.Key.fromString(keydata.publicRSA_openssh).blob()