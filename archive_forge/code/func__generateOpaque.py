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
def _generateOpaque(self, nonce, clientip):
    """
        Generate an opaque to be returned to the client.  This is a unique
        string that can be returned to us and verified.
        """
    now = b'%d' % (int(self._getTime()),)
    if not clientip:
        clientip = b''
    elif isinstance(clientip, str):
        clientip = clientip.encode('ascii')
    key = b','.join((nonce, clientip, now))
    digest = hexlify(md5(key + self.privateKey).digest())
    ekey = base64.b64encode(key)
    return b'-'.join((digest, ekey.replace(b'\n', b'')))