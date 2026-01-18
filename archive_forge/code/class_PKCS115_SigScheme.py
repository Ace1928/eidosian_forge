import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.asn1 import DerSequence, DerNull, DerOctetString, DerObjectId
class PKCS115_SigScheme:
    """A signature object for ``RSASSA-PKCS1-v1_5``.
    Do not instantiate directly.
    Use :func:`Cryptodome.Signature.pkcs1_15.new`.
    """

    def __init__(self, rsa_key):
        """Initialize this PKCS#1 v1.5 signature scheme object.

        :Parameters:
          rsa_key : an RSA key object
            Creation of signatures is only possible if this is a *private*
            RSA key. Verification of signatures is always possible.
        """
        self._key = rsa_key

    def can_sign(self):
        """Return ``True`` if this object can be used to sign messages."""
        return self._key.has_private()

    def sign(self, msg_hash):
        """Create the PKCS#1 v1.5 signature of a message.

        This function is also called ``RSASSA-PKCS1-V1_5-SIGN`` and
        it is specified in
        `section 8.2.1 of RFC8017 <https://tools.ietf.org/html/rfc8017#page-36>`_.

        :parameter msg_hash:
            This is an object from the :mod:`Cryptodome.Hash` package.
            It has been used to digest the message to sign.
        :type msg_hash: hash object

        :return: the signature encoded as a *byte string*.
        :raise ValueError: if the RSA key is not long enough for the given hash algorithm.
        :raise TypeError: if the RSA key has no private half.
        """
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        em = _EMSA_PKCS1_V1_5_ENCODE(msg_hash, k)
        em_int = bytes_to_long(em)
        signature = self._key._decrypt_to_bytes(em_int)
        if em_int != pow(bytes_to_long(signature), self._key.e, self._key.n):
            raise ValueError('Fault detected in RSA private key operation')
        return signature

    def verify(self, msg_hash, signature):
        """Check if the  PKCS#1 v1.5 signature over a message is valid.

        This function is also called ``RSASSA-PKCS1-V1_5-VERIFY`` and
        it is specified in
        `section 8.2.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#page-37>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Cryptodome.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: byte string

        :raise ValueError: if the signature is not valid.
        """
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        if len(signature) != k:
            raise ValueError('Invalid signature')
        signature_int = bytes_to_long(signature)
        em_int = self._key._encrypt(signature_int)
        em1 = long_to_bytes(em_int, k)
        try:
            possible_em1 = [_EMSA_PKCS1_V1_5_ENCODE(msg_hash, k, True)]
            try:
                algorithm_is_md = msg_hash.oid.startswith('1.2.840.113549.2.')
            except AttributeError:
                algorithm_is_md = False
            if not algorithm_is_md:
                possible_em1.append(_EMSA_PKCS1_V1_5_ENCODE(msg_hash, k, False))
        except ValueError:
            raise ValueError('Invalid signature')
        if em1 not in possible_em1:
            raise ValueError('Invalid signature')
        pass