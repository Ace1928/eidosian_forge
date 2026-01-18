from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
class FipsDsaSigScheme(DssSigScheme):
    _fips_186_3_L_N = ((1024, 160), (2048, 224), (2048, 256), (3072, 256))

    def __init__(self, key, encoding, order, randfunc):
        super(FipsDsaSigScheme, self).__init__(key, encoding, order)
        self._randfunc = randfunc
        L = Integer(key.p).size_in_bits()
        if (L, self._order_bits) not in self._fips_186_3_L_N:
            error = 'L/N (%d, %d) is not compliant to FIPS 186-3' % (L, self._order_bits)
            raise ValueError(error)

    def _compute_nonce(self, msg_hash):
        return Integer.random_range(min_inclusive=1, max_exclusive=self._order, randfunc=self._randfunc)

    def _valid_hash(self, msg_hash):
        """Verify that SHA-1, SHA-2 or SHA-3 are used"""
        return msg_hash.oid == '1.3.14.3.2.26' or msg_hash.oid.startswith('2.16.840.1.101.3.4.2.')