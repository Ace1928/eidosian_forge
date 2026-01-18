import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
@implementer(IClientAuthentication)
class CramMD5ClientAuthenticator:

    def __init__(self, user):
        self.user = user

    def getName(self):
        return b'CRAM-MD5'

    def challengeResponse(self, secret, chal):
        response = hmac.HMAC(secret, chal, digestmod=hashlib.md5).hexdigest()
        return self.user + b' ' + response.encode('ascii')