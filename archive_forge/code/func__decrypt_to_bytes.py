import binascii
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import tobytes, bord, tostr
from Cryptodome.Util.asn1 import DerSequence, DerNull
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
def _decrypt_to_bytes(self, ciphertext):
    if not 0 <= ciphertext < self._n:
        raise ValueError('Ciphertext too large')
    if not self.has_private():
        raise TypeError('This is not a private key')
    r = Integer.random_range(min_inclusive=1, max_exclusive=self._n)
    cp = Integer(ciphertext) * pow(r, self._e, self._n) % self._n
    m1 = pow(cp, self._dp, self._p)
    m2 = pow(cp, self._dq, self._q)
    h = (m2 - m1) * self._u % self._q
    mp = h * self._p + m1
    result = Integer._mult_modulo_bytes(r.inverse(self._n), mp, self._n)
    return result