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
@implementer(ICredentialsChecker)
class PrivateKeyChecker:
    """
    A very simple public key checker which authenticates anyone whose
    public/private keypair is the same keydata.public/privateRSA_openssh.
    """
    credentialInterfaces = (ISSHPrivateKey,)

    def requestAvatarId(self, creds):
        if creds.blob == keys.Key.fromString(keydata.publicRSA_openssh).blob():
            if creds.signature is not None:
                obj = keys.Key.fromString(creds.blob)
                if obj.verify(creds.signature, creds.sigData):
                    return creds.username
            else:
                raise ValidPublicKey()
        raise UnauthorizedLogin()