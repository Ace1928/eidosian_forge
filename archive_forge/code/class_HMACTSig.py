import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
class HMACTSig:
    """
    HMAC TSIG implementation.  This uses the HMAC python module to handle the
    sign/verify operations.
    """
    _hashes = {HMAC_SHA1: hashlib.sha1, HMAC_SHA224: hashlib.sha224, HMAC_SHA256: hashlib.sha256, HMAC_SHA256_128: (hashlib.sha256, 128), HMAC_SHA384: hashlib.sha384, HMAC_SHA384_192: (hashlib.sha384, 192), HMAC_SHA512: hashlib.sha512, HMAC_SHA512_256: (hashlib.sha512, 256), HMAC_MD5: hashlib.md5}

    def __init__(self, key, algorithm):
        try:
            hashinfo = self._hashes[algorithm]
        except KeyError:
            raise NotImplementedError(f'TSIG algorithm {algorithm} is not supported')
        if isinstance(hashinfo, tuple):
            self.hmac_context = hmac.new(key, digestmod=hashinfo[0])
            self.size = hashinfo[1]
        else:
            self.hmac_context = hmac.new(key, digestmod=hashinfo)
            self.size = None
        self.name = self.hmac_context.name
        if self.size:
            self.name += f'-{self.size}'

    def update(self, data):
        return self.hmac_context.update(data)

    def sign(self):
        digest = self.hmac_context.digest()
        if self.size:
            digest = digest[:self.size // 8]
        return digest

    def verify(self, expected):
        mac = self.sign()
        if not hmac.compare_digest(mac, expected):
            raise BadSignature