import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
class NetscapeSPKI:
    """
    A Netscape SPKI object.
    """

    def __init__(self) -> None:
        spki = _lib.NETSCAPE_SPKI_new()
        self._spki = _ffi.gc(spki, _lib.NETSCAPE_SPKI_free)

    def sign(self, pkey: PKey, digest: str) -> None:
        """
        Sign the certificate request with this key and digest type.

        :param pkey: The private key to sign with.
        :type pkey: :py:class:`PKey`

        :param digest: The message digest to use.
        :type digest: :py:class:`str`

        :return: ``None``
        """
        if pkey._only_public:
            raise ValueError('Key has only public part')
        if not pkey._initialized:
            raise ValueError('Key is uninitialized')
        digest_obj = _lib.EVP_get_digestbyname(_byte_string(digest))
        if digest_obj == _ffi.NULL:
            raise ValueError('No such digest method')
        sign_result = _lib.NETSCAPE_SPKI_sign(self._spki, pkey._pkey, digest_obj)
        _openssl_assert(sign_result > 0)

    def verify(self, key: PKey) -> bool:
        """
        Verifies a signature on a certificate request.

        :param PKey key: The public key that signature is supposedly from.

        :return: ``True`` if the signature is correct.
        :rtype: bool

        :raises OpenSSL.crypto.Error: If the signature is invalid, or there was
            a problem verifying the signature.
        """
        answer = _lib.NETSCAPE_SPKI_verify(self._spki, key._pkey)
        if answer <= 0:
            _raise_current_error()
        return True

    def b64_encode(self) -> bytes:
        """
        Generate a base64 encoded representation of this SPKI object.

        :return: The base64 encoded string.
        :rtype: :py:class:`bytes`
        """
        encoded = _lib.NETSCAPE_SPKI_b64_encode(self._spki)
        result = _ffi.string(encoded)
        _lib.OPENSSL_free(encoded)
        return result

    def get_pubkey(self) -> PKey:
        """
        Get the public key of this certificate.

        :return: The public key.
        :rtype: :py:class:`PKey`
        """
        pkey = PKey.__new__(PKey)
        pkey._pkey = _lib.NETSCAPE_SPKI_get_pubkey(self._spki)
        _openssl_assert(pkey._pkey != _ffi.NULL)
        pkey._pkey = _ffi.gc(pkey._pkey, _lib.EVP_PKEY_free)
        pkey._only_public = True
        return pkey

    def set_pubkey(self, pkey: PKey) -> None:
        """
        Set the public key of the certificate

        :param pkey: The public key
        :return: ``None``
        """
        set_result = _lib.NETSCAPE_SPKI_set_pubkey(self._spki, pkey._pkey)
        _openssl_assert(set_result == 1)