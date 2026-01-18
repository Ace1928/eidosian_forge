from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
def _int2octets(self, int_mod_q):
    """See 2.3.3 in RFC6979"""
    assert 0 < int_mod_q < self._order
    return long_to_bytes(int_mod_q, self._order_bytes)