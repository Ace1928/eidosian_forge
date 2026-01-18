from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
def _verify_ed448(self, msg_or_hash, signature, ph):
    if len(signature) != 114:
        raise ValueError('The signature is not authentic (length)')
    flag = int(ph)
    dom4 = b'SigEd448' + bchr(flag) + bchr(len(self._context)) + self._context
    PHM = msg_or_hash.read(64) if ph else msg_or_hash
    try:
        R = import_public_key(signature[:57]).pointQ
    except ValueError:
        raise ValueError('The signature is not authentic (R)')
    s = Integer.from_bytes(signature[57:], 'little')
    if s > self._order:
        raise ValueError('The signature is not authentic (S)')
    k_hash = SHAKE256.new(dom4 + signature[:57] + self._A + PHM).read(114)
    k = Integer.from_bytes(k_hash, 'little') % self._order
    point1 = s * 8 * self._key._curve.G
    point2 = 8 * R + k * 8 * self._key.pointQ
    if point1 != point2:
        raise ValueError('The signature is not authentic')