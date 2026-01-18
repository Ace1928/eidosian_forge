from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
class PSS_SigScheme:
    """A signature object for ``RSASSA-PSS``.
    Do not instantiate directly.
    Use :func:`Cryptodome.Signature.pss.new`.
    """

    def __init__(self, key, mgfunc, saltLen, randfunc):
        """Initialize this PKCS#1 PSS signature scheme object.

        :Parameters:
          key : an RSA key object
            If a private half is given, both signature and
            verification are possible.
            If a public half is given, only verification is possible.
          mgfunc : callable
            A mask generation function that accepts two parameters:
            a string to use as seed, and the lenth of the mask to
            generate, in bytes.
          saltLen : integer
            Length of the salt, in bytes.
          randfunc : callable
            A function that returns random bytes.
        """
        self._key = key
        self._saltLen = saltLen
        self._mgfunc = mgfunc
        self._randfunc = randfunc

    def can_sign(self):
        """Return ``True`` if this object can be used to sign messages."""
        return self._key.has_private()

    def sign(self, msg_hash):
        """Create the PKCS#1 PSS signature of a message.

        This function is also called ``RSASSA-PSS-SIGN`` and
        it is specified in
        `section 8.1.1 of RFC8017 <https://tools.ietf.org/html/rfc8017#section-8.1.1>`_.

        :parameter msg_hash:
            This is an object from the :mod:`Cryptodome.Hash` package.
            It has been used to digest the message to sign.
        :type msg_hash: hash object

        :return: the signature encoded as a *byte string*.
        :raise ValueError: if the RSA key is not long enough for the given hash algorithm.
        :raise TypeError: if the RSA key has no private half.
        """
        if self._saltLen is None:
            sLen = msg_hash.digest_size
        else:
            sLen = self._saltLen
        if self._mgfunc is None:
            mgf = lambda x, y: MGF1(x, y, msg_hash)
        else:
            mgf = self._mgfunc
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        em = _EMSA_PSS_ENCODE(msg_hash, modBits - 1, self._randfunc, mgf, sLen)
        em_int = bytes_to_long(em)
        signature = self._key._decrypt_to_bytes(em_int)
        if em_int != pow(bytes_to_long(signature), self._key.e, self._key.n):
            raise ValueError('Fault detected in RSA private key operation')
        return signature

    def verify(self, msg_hash, signature):
        """Check if the  PKCS#1 PSS signature over a message is valid.

        This function is also called ``RSASSA-PSS-VERIFY`` and
        it is specified in
        `section 8.1.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#section-8.1.2>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Cryptodome.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: bytes

        :raise ValueError: if the signature is not valid.
        """
        if self._saltLen is None:
            sLen = msg_hash.digest_size
        else:
            sLen = self._saltLen
        if self._mgfunc:
            mgf = self._mgfunc
        else:
            mgf = lambda x, y: MGF1(x, y, msg_hash)
        modBits = Cryptodome.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        if len(signature) != k:
            raise ValueError('Incorrect signature')
        signature_int = bytes_to_long(signature)
        em_int = self._key._encrypt(signature_int)
        emLen = ceil_div(modBits - 1, 8)
        em = long_to_bytes(em_int, emLen)
        _EMSA_PSS_VERIFY(msg_hash, em, modBits - 1, mgf, sLen)