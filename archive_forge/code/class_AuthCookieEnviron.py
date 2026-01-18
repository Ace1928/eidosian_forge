import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
class AuthCookieEnviron(list):
    """
    a list of environment keys to be saved via cookie

    An instance of this object, found at ``environ['paste.auth.cookie']``
    lists the `environ` keys that were restored from or will be added
    to the digially signed cookie.  This object can be accessed from an
    `environ` variable by using this module's name.
    """

    def __init__(self, handler, scanlist):
        list.__init__(self, scanlist)
        self.handler = handler

    def append(self, value):
        if value in self:
            return
        list.append(self, str(value))