import types
from Cryptodome.Signature import pss
def _pycrypto_verify(self, hash_object, signature):
    try:
        self._verify(hash_object, signature)
    except (ValueError, TypeError):
        return False
    return True