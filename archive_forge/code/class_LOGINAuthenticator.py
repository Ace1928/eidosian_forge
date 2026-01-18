import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
@implementer(IClientAuthentication)
class LOGINAuthenticator:

    def __init__(self, user):
        self.user = user
        self.challengeResponse = self.challengeUsername

    def getName(self):
        return b'LOGIN'

    def challengeUsername(self, secret, chal):
        self.challengeResponse = self.challengeSecret
        return self.user

    def challengeSecret(self, secret, chal):
        return secret