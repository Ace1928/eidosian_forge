import base64
import hmac
import random
import re
import time
from binascii import hexlify
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred import error
from twisted.cred._digest import calcHA1, calcHA2, calcResponse
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.randbytes import secureRandom
from twisted.python.versions import Version
@implementer(IUsernameHashedPassword)
class CramMD5Credentials:
    """
    An encapsulation of some CramMD5 hashed credentials.

    @ivar challenge: The challenge to be sent to the client.
    @type challenge: L{bytes}

    @ivar response: The hashed response from the client.
    @type response: L{bytes}

    @ivar username: The username from the response from the client.
    @type username: L{bytes} or L{None} if not yet provided.
    """
    username = None
    challenge = b''
    response = b''

    def __init__(self, host=None):
        self.host = host

    def getChallenge(self):
        if self.challenge:
            return self.challenge
        r = random.randrange(2147483647)
        t = time.time()
        self.challenge = networkString('<%d.%d@%s>' % (r, t, nativeString(self.host) if self.host else None))
        return self.challenge

    def setResponse(self, response):
        self.username, self.response = response.split(None, 1)

    def moreChallenges(self):
        return False

    def checkPassword(self, password):
        verify = hexlify(hmac.HMAC(password, self.challenge, digestmod=md5).digest())
        return verify == self.response