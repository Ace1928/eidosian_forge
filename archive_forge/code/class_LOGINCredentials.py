import hashlib
import hmac
from zope.interface import implementer
from twisted.cred import credentials
from twisted.mail._except import IllegalClientResponse
from twisted.mail.interfaces import IChallengeResponse, IClientAuthentication
from twisted.python.compat import nativeString
@implementer(IChallengeResponse)
class LOGINCredentials(credentials.UsernamePassword):

    def __init__(self):
        self.challenges = [b'Password\x00', b'User Name\x00']
        self.responses = [b'password', b'username']
        credentials.UsernamePassword.__init__(self, None, None)

    def getChallenge(self):
        return self.challenges.pop()

    def setResponse(self, response):
        setattr(self, nativeString(self.responses.pop()), response)

    def moreChallenges(self):
        return bool(self.challenges)