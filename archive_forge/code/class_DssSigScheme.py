from Cryptodome.Util.asn1 import DerSequence
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import HMAC
from Cryptodome.PublicKey.ECC import EccKey
from Cryptodome.PublicKey.DSA import DsaKey
class DssSigScheme(object):
    """A (EC)DSA signature object.
    Do not instantiate directly.
    Use :func:`Cryptodome.Signature.DSS.new`.
    """

    def __init__(self, key, encoding, order):
        """Create a new Digital Signature Standard (DSS) object.

        Do not instantiate this object directly,
        use `Cryptodome.Signature.DSS.new` instead.
        """
        self._key = key
        self._encoding = encoding
        self._order = order
        self._order_bits = self._order.size_in_bits()
        self._order_bytes = (self._order_bits - 1) // 8 + 1

    def can_sign(self):
        """Return ``True`` if this signature object can be used
        for signing messages."""
        return self._key.has_private()

    def _compute_nonce(self, msg_hash):
        raise NotImplementedError('To be provided by subclasses')

    def _valid_hash(self, msg_hash):
        raise NotImplementedError('To be provided by subclasses')

    def sign(self, msg_hash):
        """Compute the DSA/ECDSA signature of a message.

        Args:
          msg_hash (hash object):
            The hash that was carried out over the message.
            The object belongs to the :mod:`Cryptodome.Hash` package.
            Under mode ``'fips-186-3'``, the hash must be a FIPS
            approved secure hash (SHA-2 or SHA-3).

        :return: The signature as ``bytes``
        :raise ValueError: if the hash algorithm is incompatible to the (EC)DSA key
        :raise TypeError: if the (EC)DSA key has no private half
        """
        if not self._key.has_private():
            raise TypeError('Private key is needed to sign')
        if not self._valid_hash(msg_hash):
            raise ValueError('Hash is not sufficiently strong')
        nonce = self._compute_nonce(msg_hash)
        z = Integer.from_bytes(msg_hash.digest()[:self._order_bytes])
        sig_pair = self._key._sign(z, nonce)
        if self._encoding == 'binary':
            output = b''.join([long_to_bytes(x, self._order_bytes) for x in sig_pair])
        else:
            output = DerSequence(sig_pair).encode()
        return output

    def verify(self, msg_hash, signature):
        """Check if a certain (EC)DSA signature is authentic.

        Args:
          msg_hash (hash object):
            The hash that was carried out over the message.
            This is an object belonging to the :mod:`Cryptodome.Hash` module.
            Under mode ``'fips-186-3'``, the hash must be a FIPS
            approved secure hash (SHA-2 or SHA-3).

          signature (``bytes``):
            The signature that needs to be validated.

        :raise ValueError: if the signature is not authentic
        """
        if not self._valid_hash(msg_hash):
            raise ValueError('Hash is not sufficiently strong')
        if self._encoding == 'binary':
            if len(signature) != 2 * self._order_bytes:
                raise ValueError('The signature is not authentic (length)')
            r_prime, s_prime = [Integer.from_bytes(x) for x in (signature[:self._order_bytes], signature[self._order_bytes:])]
        else:
            try:
                der_seq = DerSequence().decode(signature, strict=True)
            except (ValueError, IndexError):
                raise ValueError('The signature is not authentic (DER)')
            if len(der_seq) != 2 or not der_seq.hasOnlyInts():
                raise ValueError('The signature is not authentic (DER content)')
            r_prime, s_prime = (Integer(der_seq[0]), Integer(der_seq[1]))
        if not 0 < r_prime < self._order or not 0 < s_prime < self._order:
            raise ValueError('The signature is not authentic (d)')
        z = Integer.from_bytes(msg_hash.digest()[:self._order_bytes])
        result = self._key._verify(z, (r_prime, s_prime))
        if not result:
            raise ValueError('The signature is not authentic')
        return False