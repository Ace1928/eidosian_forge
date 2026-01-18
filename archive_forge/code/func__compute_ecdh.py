from Cryptodome.Util.number import long_to_bytes
from Cryptodome.PublicKey.ECC import EccKey
def _compute_ecdh(key_priv, key_pub):
    pointP = key_pub.pointQ * key_priv.d
    if pointP.is_point_at_infinity():
        raise ValueError('Invalid ECDH point')
    z = long_to_bytes(pointP.x, pointP.size_in_bytes())
    return z