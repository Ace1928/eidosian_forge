import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
@implementer(cred.credentials.IUsernameHashedPassword)
class APOPCredentials:
    """
    Credentials for use in APOP authentication.

    @ivar magic: See L{__init__}
    @ivar username: See L{__init__}
    @ivar digest: See L{__init__}
    """

    def __init__(self, magic, username, digest):
        """
        @type magic: L{bytes}
        @param magic: The challenge string used to encrypt the password.

        @type username: L{bytes}
        @param username: The username associated with these credentials.

        @type digest: L{bytes}
        @param digest: An encrypted version of the user's password.  Should be
            generated as an MD5 hash of the challenge string concatenated with
            the plaintext password.
        """
        self.magic = magic
        self.username = username
        self.digest = digest

    def checkPassword(self, password):
        """
        Validate a plaintext password against the credentials.

        @type password: L{bytes}
        @param password: A plaintext password.

        @rtype: L{bool}
        @return: C{True} if the credentials represented by this object match
        the given password, C{False} if they do not.
        """
        seed = self.magic + password
        myDigest = md5(seed).hexdigest()
        return myDigest == self.digest