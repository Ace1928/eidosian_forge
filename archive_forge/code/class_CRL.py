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
class CRL:
    """
    A certificate revocation list.
    """

    def __init__(self) -> None:
        crl = _lib.X509_CRL_new()
        self._crl = _ffi.gc(crl, _lib.X509_CRL_free)

    def to_cryptography(self) -> x509.CertificateRevocationList:
        """
        Export as a ``cryptography`` CRL.

        :rtype: ``cryptography.x509.CertificateRevocationList``

        .. versionadded:: 17.1.0
        """
        from cryptography.x509 import load_der_x509_crl
        der = dump_crl(FILETYPE_ASN1, self)
        return load_der_x509_crl(der)

    @classmethod
    def from_cryptography(cls, crypto_crl: x509.CertificateRevocationList) -> 'CRL':
        """
        Construct based on a ``cryptography`` *crypto_crl*.

        :param crypto_crl: A ``cryptography`` certificate revocation list
        :type crypto_crl: ``cryptography.x509.CertificateRevocationList``

        :rtype: CRL

        .. versionadded:: 17.1.0
        """
        if not isinstance(crypto_crl, x509.CertificateRevocationList):
            raise TypeError('Must be a certificate revocation list')
        from cryptography.hazmat.primitives.serialization import Encoding
        der = crypto_crl.public_bytes(Encoding.DER)
        return load_crl(FILETYPE_ASN1, der)

    def get_revoked(self) -> Optional[Tuple[Revoked, ...]]:
        """
        Return the revocations in this certificate revocation list.

        These revocations will be provided by value, not by reference.
        That means it's okay to mutate them: it won't affect this CRL.

        :return: The revocations in this CRL.
        :rtype: :class:`tuple` of :class:`Revocation`
        """
        results = []
        revoked_stack = _lib.X509_CRL_get_REVOKED(self._crl)
        for i in range(_lib.sk_X509_REVOKED_num(revoked_stack)):
            revoked = _lib.sk_X509_REVOKED_value(revoked_stack, i)
            revoked_copy = _lib.X509_REVOKED_dup(revoked)
            pyrev = Revoked.__new__(Revoked)
            pyrev._revoked = _ffi.gc(revoked_copy, _lib.X509_REVOKED_free)
            results.append(pyrev)
        if results:
            return tuple(results)
        return None

    def add_revoked(self, revoked: Revoked) -> None:
        """
        Add a revoked (by value not reference) to the CRL structure

        This revocation will be added by value, not by reference. That
        means it's okay to mutate it after adding: it won't affect
        this CRL.

        :param Revoked revoked: The new revocation.
        :return: ``None``
        """
        copy = _lib.X509_REVOKED_dup(revoked._revoked)
        _openssl_assert(copy != _ffi.NULL)
        add_result = _lib.X509_CRL_add0_revoked(self._crl, copy)
        _openssl_assert(add_result != 0)

    def get_issuer(self) -> X509Name:
        """
        Get the CRL's issuer.

        .. versionadded:: 16.1.0

        :rtype: X509Name
        """
        _issuer = _lib.X509_NAME_dup(_lib.X509_CRL_get_issuer(self._crl))
        _openssl_assert(_issuer != _ffi.NULL)
        _issuer = _ffi.gc(_issuer, _lib.X509_NAME_free)
        issuer = X509Name.__new__(X509Name)
        issuer._name = _issuer
        return issuer

    def set_version(self, version: int) -> None:
        """
        Set the CRL version.

        .. versionadded:: 16.1.0

        :param int version: The version of the CRL.
        :return: ``None``
        """
        _openssl_assert(_lib.X509_CRL_set_version(self._crl, version) != 0)

    def set_lastUpdate(self, when: bytes) -> None:
        """
        Set when the CRL was last updated.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        .. versionadded:: 16.1.0

        :param bytes when: A timestamp string.
        :return: ``None``
        """
        lastUpdate = _new_asn1_time(when)
        ret = _lib.X509_CRL_set1_lastUpdate(self._crl, lastUpdate)
        _openssl_assert(ret == 1)

    def set_nextUpdate(self, when: bytes) -> None:
        """
        Set when the CRL will next be updated.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        .. versionadded:: 16.1.0

        :param bytes when: A timestamp string.
        :return: ``None``
        """
        nextUpdate = _new_asn1_time(when)
        ret = _lib.X509_CRL_set1_nextUpdate(self._crl, nextUpdate)
        _openssl_assert(ret == 1)

    def sign(self, issuer_cert: X509, issuer_key: PKey, digest: bytes) -> None:
        """
        Sign the CRL.

        Signing a CRL enables clients to associate the CRL itself with an
        issuer. Before a CRL is meaningful to other OpenSSL functions, it must
        be signed by an issuer.

        This method implicitly sets the issuer's name based on the issuer
        certificate and private key used to sign the CRL.

        .. versionadded:: 16.1.0

        :param X509 issuer_cert: The issuer's certificate.
        :param PKey issuer_key: The issuer's private key.
        :param bytes digest: The digest method to sign the CRL with.
        """
        digest_obj = _lib.EVP_get_digestbyname(digest)
        _openssl_assert(digest_obj != _ffi.NULL)
        _lib.X509_CRL_set_issuer_name(self._crl, _lib.X509_get_subject_name(issuer_cert._x509))
        _lib.X509_CRL_sort(self._crl)
        result = _lib.X509_CRL_sign(self._crl, issuer_key._pkey, digest_obj)
        _openssl_assert(result != 0)

    def export(self, cert: X509, key: PKey, type: int=FILETYPE_PEM, days: int=100, digest: bytes=_UNSPECIFIED) -> bytes:
        """
        Export the CRL as a string.

        :param X509 cert: The certificate used to sign the CRL.
        :param PKey key: The key used to sign the CRL.
        :param int type: The export format, either :data:`FILETYPE_PEM`,
            :data:`FILETYPE_ASN1`, or :data:`FILETYPE_TEXT`.
        :param int days: The number of days until the next update of this CRL.
        :param bytes digest: The name of the message digest to use (eg
            ``b"sha256"``).
        :rtype: bytes
        """
        if not isinstance(cert, X509):
            raise TypeError('cert must be an X509 instance')
        if not isinstance(key, PKey):
            raise TypeError('key must be a PKey instance')
        if not isinstance(type, int):
            raise TypeError('type must be an integer')
        if digest is _UNSPECIFIED:
            raise TypeError('digest must be provided')
        digest_obj = _lib.EVP_get_digestbyname(digest)
        if digest_obj == _ffi.NULL:
            raise ValueError('No such digest method')
        sometime = _lib.ASN1_TIME_new()
        _openssl_assert(sometime != _ffi.NULL)
        sometime = _ffi.gc(sometime, _lib.ASN1_TIME_free)
        ret = _lib.X509_gmtime_adj(sometime, 0)
        _openssl_assert(ret != _ffi.NULL)
        ret = _lib.X509_CRL_set1_lastUpdate(self._crl, sometime)
        _openssl_assert(ret == 1)
        ret = _lib.X509_gmtime_adj(sometime, days * 24 * 60 * 60)
        _openssl_assert(ret != _ffi.NULL)
        ret = _lib.X509_CRL_set1_nextUpdate(self._crl, sometime)
        _openssl_assert(ret == 1)
        ret = _lib.X509_CRL_set_issuer_name(self._crl, _lib.X509_get_subject_name(cert._x509))
        _openssl_assert(ret == 1)
        sign_result = _lib.X509_CRL_sign(self._crl, key._pkey, digest_obj)
        if not sign_result:
            _raise_current_error()
        return dump_crl(type, self)