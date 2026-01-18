from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
def _sign_ed25519(self, msg_or_hash, ph):
    if self._context or ph:
        flag = int(ph)
        dom2 = b'SigEd25519 no Ed25519 collisions' + bchr(flag) + bchr(len(self._context)) + self._context
    else:
        dom2 = b''
    PHM = msg_or_hash.digest() if ph else msg_or_hash
    r_hash = SHA512.new(dom2 + self._key._prefix + PHM).digest()
    r = Integer.from_bytes(r_hash, 'little') % self._order
    R_pk = EccKey(point=r * self._key._curve.G)._export_eddsa()
    k_hash = SHA512.new(dom2 + R_pk + self._A + PHM).digest()
    k = Integer.from_bytes(k_hash, 'little') % self._order
    s = (r + k * self._key.d) % self._order
    return R_pk + s.to_bytes(32, 'little')