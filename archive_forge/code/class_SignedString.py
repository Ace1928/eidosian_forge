import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class SignedString(FancyValidator):
    """
    Encodes a string into a signed string, and base64 encodes both the
    signature string and a random nonce.

    It is up to you to provide a secret, and to keep the secret handy
    and consistent.
    """
    messages = dict(malformed=_('Value does not contain a signature'), badsig=_('Signature is not correct'))
    secret = None
    nonce_length = 4

    def _convert_to_python(self, value, state):
        global sha1
        if not sha1:
            from hashlib import sha1
        assert self.secret is not None, 'You must give a secret'
        parts = value.split(None, 1)
        if not parts or len(parts) == 1:
            raise Invalid(self.message('malformed', state), value, state)
        sig, rest = parts
        sig = sig.decode('base64')
        rest = rest.decode('base64')
        nonce = rest[:self.nonce_length]
        rest = rest[self.nonce_length:]
        expected = sha1(str(self.secret) + nonce + rest).digest()
        if expected != sig:
            raise Invalid(self.message('badsig', state), value, state)
        return rest

    def _convert_from_python(self, value, state):
        global sha1
        if not sha1:
            from hashlib import sha1
        nonce = self.make_nonce()
        value = str(value)
        digest = sha1(self.secret + nonce + value).digest()
        return self.encode(digest) + ' ' + self.encode(nonce + value)

    def encode(self, value):
        return value.encode('base64').strip().replace('\n', '')

    def make_nonce(self):
        global random
        if not random:
            import random
        return ''.join((chr(random.randrange(256)) for _i in range(self.nonce_length)))