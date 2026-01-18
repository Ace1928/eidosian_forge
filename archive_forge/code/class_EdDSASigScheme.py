from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
class EdDSASigScheme(object):
    """An EdDSA signature object.
    Do not instantiate directly.
    Use :func:`Cryptodome.Signature.eddsa.new`.
    """

    def __init__(self, key, context):
        """Create a new EdDSA object.

        Do not instantiate this object directly,
        use `Cryptodome.Signature.DSS.new` instead.
        """
        self._key = key
        self._context = context
        self._A = key._export_eddsa()
        self._order = key._curve.order

    def can_sign(self):
        """Return ``True`` if this signature object can be used
        for signing messages."""
        return self._key.has_private()

    def sign(self, msg_or_hash):
        """Compute the EdDSA signature of a message.

        Args:
          msg_or_hash (bytes or a hash object):
            The message to sign (``bytes``, in case of *PureEdDSA*) or
            the hash that was carried out over the message (hash object, for *HashEdDSA*).

            The hash object must be :class:`Cryptodome.Hash.SHA512` for Ed25519,
            and :class:`Cryptodome.Hash.SHAKE256` object for Ed448.

        :return: The signature as ``bytes``. It is always 64 bytes for Ed25519, and 114 bytes for Ed448.
        :raise TypeError: if the EdDSA key has no private half
        """
        if not self._key.has_private():
            raise TypeError('Private key is needed to sign')
        if self._key._curve.name == 'ed25519':
            ph = isinstance(msg_or_hash, SHA512.SHA512Hash)
            if not (ph or is_bytes(msg_or_hash)):
                raise TypeError("'msg_or_hash' must be bytes of a SHA-512 hash")
            eddsa_sign_method = self._sign_ed25519
        elif self._key._curve.name == 'ed448':
            ph = isinstance(msg_or_hash, SHAKE256.SHAKE256_XOF)
            if not (ph or is_bytes(msg_or_hash)):
                raise TypeError("'msg_or_hash' must be bytes of a SHAKE256 hash")
            eddsa_sign_method = self._sign_ed448
        else:
            raise ValueError('Incorrect curve for EdDSA')
        return eddsa_sign_method(msg_or_hash, ph)

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

    def _sign_ed448(self, msg_or_hash, ph):
        flag = int(ph)
        dom4 = b'SigEd448' + bchr(flag) + bchr(len(self._context)) + self._context
        PHM = msg_or_hash.read(64) if ph else msg_or_hash
        r_hash = SHAKE256.new(dom4 + self._key._prefix + PHM).read(114)
        r = Integer.from_bytes(r_hash, 'little') % self._order
        R_pk = EccKey(point=r * self._key._curve.G)._export_eddsa()
        k_hash = SHAKE256.new(dom4 + R_pk + self._A + PHM).read(114)
        k = Integer.from_bytes(k_hash, 'little') % self._order
        s = (r + k * self._key.d) % self._order
        return R_pk + s.to_bytes(57, 'little')

    def verify(self, msg_or_hash, signature):
        """Check if an EdDSA signature is authentic.

        Args:
          msg_or_hash (bytes or a hash object):
            The message to verify (``bytes``, in case of *PureEdDSA*) or
            the hash that was carried out over the message (hash object, for *HashEdDSA*).

            The hash object must be :class:`Cryptodome.Hash.SHA512` object for Ed25519,
            and :class:`Cryptodome.Hash.SHAKE256` for Ed448.

          signature (``bytes``):
            The signature that needs to be validated.
            It must be 64 bytes for Ed25519, and 114 bytes for Ed448.

        :raise ValueError: if the signature is not authentic
        """
        if self._key._curve.name == 'ed25519':
            ph = isinstance(msg_or_hash, SHA512.SHA512Hash)
            if not (ph or is_bytes(msg_or_hash)):
                raise TypeError("'msg_or_hash' must be bytes of a SHA-512 hash")
            eddsa_verify_method = self._verify_ed25519
        elif self._key._curve.name == 'ed448':
            ph = isinstance(msg_or_hash, SHAKE256.SHAKE256_XOF)
            if not (ph or is_bytes(msg_or_hash)):
                raise TypeError("'msg_or_hash' must be bytes of a SHAKE256 hash")
            eddsa_verify_method = self._verify_ed448
        else:
            raise ValueError('Incorrect curve for EdDSA')
        return eddsa_verify_method(msg_or_hash, signature, ph)

    def _verify_ed25519(self, msg_or_hash, signature, ph):
        if len(signature) != 64:
            raise ValueError('The signature is not authentic (length)')
        if self._context or ph:
            flag = int(ph)
            dom2 = b'SigEd25519 no Ed25519 collisions' + bchr(flag) + bchr(len(self._context)) + self._context
        else:
            dom2 = b''
        PHM = msg_or_hash.digest() if ph else msg_or_hash
        try:
            R = import_public_key(signature[:32]).pointQ
        except ValueError:
            raise ValueError('The signature is not authentic (R)')
        s = Integer.from_bytes(signature[32:], 'little')
        if s > self._order:
            raise ValueError('The signature is not authentic (S)')
        k_hash = SHA512.new(dom2 + signature[:32] + self._A + PHM).digest()
        k = Integer.from_bytes(k_hash, 'little') % self._order
        point1 = s * 8 * self._key._curve.G
        point2 = 8 * R + k * 8 * self._key.pointQ
        if point1 != point2:
            raise ValueError('The signature is not authentic')

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