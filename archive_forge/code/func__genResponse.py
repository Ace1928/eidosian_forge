import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def _genResponse(self, charset, realm, nonce):
    """
        Generate response-value.

        Creates a response to a challenge according to section 2.1.2.1 of
        RFC 2831 using the C{charset}, C{realm} and C{nonce} directives
        from the challenge.
        """
    try:
        username = self.username.encode(charset)
        password = self.password.encode(charset)
        digest_uri = self.digest_uri.encode(charset)
    except UnicodeError:
        raise
    nc = networkString(f'{1:08x}')
    cnonce = self._gen_nonce()
    qop = b'auth'
    response = self._calculateResponse(cnonce, nc, nonce, username, password, realm, digest_uri)
    directives = {b'username': username, b'realm': realm, b'nonce': nonce, b'cnonce': cnonce, b'nc': nc, b'qop': qop, b'digest-uri': digest_uri, b'response': response, b'charset': charset.encode('ascii')}
    return self._unparse(directives)