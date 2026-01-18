import warnings as _warnings
import hashlib as _hashlib
def _init_hmac(self, key, msg, digestmod):
    self._hmac = _hashopenssl.hmac_new(key, msg, digestmod=digestmod)
    self.digest_size = self._hmac.digest_size
    self.block_size = self._hmac.block_size