import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
class FakeDigestCredentialFactory(DigestCredentialFactory):
    """
    A Fake Digest Credential Factory that generates a predictable
    nonce and opaque
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privateKey = b'0'

    def _generateNonce(self):
        """
        Generate a static nonce
        """
        return b'178288758716122392881254770685'

    def _getTime(self):
        """
        Return a stable time
        """
        return 0