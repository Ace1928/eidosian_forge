import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
@implementer(IClientAuthentication)
class PLAINAuthenticator:

    def __init__(self, user):
        self.user = user

    def getName(self):
        return b'PLAIN'

    def challengeResponse(self, secret, chal):
        return b'\x00' + self.user + b'\x00' + secret