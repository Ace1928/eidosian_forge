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
class X509:
    """
    An X.509 certificate.
    """

    def __init__(self) -> None:
        x509 = _lib.X509_new()
        _openssl_assert(x509 != _ffi.NULL)
        self._x509 = _ffi.gc(x509, _lib.X509_free)
        self._issuer_invalidator = _X509NameInvalidator()
        self._subject_invalidator = _X509NameInvalidator()

    @classmethod
    def _from_raw_x509_ptr(cls, x509: Any) -> 'X509':
        cert = cls.__new__(cls)
        cert._x509 = _ffi.gc(x509, _lib.X509_free)
        cert._issuer_invalidator = _X509NameInvalidator()
        cert._subject_invalidator = _X509NameInvalidator()
        return cert

    def to_cryptography(self) -> x509.Certificate:
        """
        Export as a ``cryptography`` certificate.

        :rtype: ``cryptography.x509.Certificate``

        .. versionadded:: 17.1.0
        """
        from cryptography.x509 import load_der_x509_certificate
        der = dump_certificate(FILETYPE_ASN1, self)
        return load_der_x509_certificate(der)

    @classmethod
    def from_cryptography(cls, crypto_cert: x509.Certificate) -> 'X509':
        """
        Construct based on a ``cryptography`` *crypto_cert*.

        :param crypto_key: A ``cryptography`` X.509 certificate.
        :type crypto_key: ``cryptography.x509.Certificate``

        :rtype: X509

        .. versionadded:: 17.1.0
        """
        if not isinstance(crypto_cert, x509.Certificate):
            raise TypeError('Must be a certificate')
        from cryptography.hazmat.primitives.serialization import Encoding
        der = crypto_cert.public_bytes(Encoding.DER)
        return load_certificate(FILETYPE_ASN1, der)

    def set_version(self, version: int) -> None:
        """
        Set the version number of the certificate. Note that the
        version value is zero-based, eg. a value of 0 is V1.

        :param version: The version number of the certificate.
        :type version: :py:class:`int`

        :return: ``None``
        """
        if not isinstance(version, int):
            raise TypeError('version must be an integer')
        _openssl_assert(_lib.X509_set_version(self._x509, version) == 1)

    def get_version(self) -> int:
        """
        Return the version number of the certificate.

        :return: The version number of the certificate.
        :rtype: :py:class:`int`
        """
        return _lib.X509_get_version(self._x509)

    def get_pubkey(self) -> PKey:
        """
        Get the public key of the certificate.

        :return: The public key.
        :rtype: :py:class:`PKey`
        """
        pkey = PKey.__new__(PKey)
        pkey._pkey = _lib.X509_get_pubkey(self._x509)
        if pkey._pkey == _ffi.NULL:
            _raise_current_error()
        pkey._pkey = _ffi.gc(pkey._pkey, _lib.EVP_PKEY_free)
        pkey._only_public = True
        return pkey

    def set_pubkey(self, pkey: PKey) -> None:
        """
        Set the public key of the certificate.

        :param pkey: The public key.
        :type pkey: :py:class:`PKey`

        :return: :py:data:`None`
        """
        if not isinstance(pkey, PKey):
            raise TypeError('pkey must be a PKey instance')
        set_result = _lib.X509_set_pubkey(self._x509, pkey._pkey)
        _openssl_assert(set_result == 1)

    def sign(self, pkey: PKey, digest: str) -> None:
        """
        Sign the certificate with this key and digest type.

        :param pkey: The key to sign with.
        :type pkey: :py:class:`PKey`

        :param digest: The name of the message digest to use.
        :type digest: :py:class:`str`

        :return: :py:data:`None`
        """
        if not isinstance(pkey, PKey):
            raise TypeError('pkey must be a PKey instance')
        if pkey._only_public:
            raise ValueError('Key only has public part')
        if not pkey._initialized:
            raise ValueError('Key is uninitialized')
        evp_md = _lib.EVP_get_digestbyname(_byte_string(digest))
        if evp_md == _ffi.NULL:
            raise ValueError('No such digest method')
        sign_result = _lib.X509_sign(self._x509, pkey._pkey, evp_md)
        _openssl_assert(sign_result > 0)

    def get_signature_algorithm(self) -> bytes:
        """
        Return the signature algorithm used in the certificate.

        :return: The name of the algorithm.
        :rtype: :py:class:`bytes`

        :raises ValueError: If the signature algorithm is undefined.

        .. versionadded:: 0.13
        """
        algor = _lib.X509_get0_tbs_sigalg(self._x509)
        nid = _lib.OBJ_obj2nid(algor.algorithm)
        if nid == _lib.NID_undef:
            raise ValueError('Undefined signature algorithm')
        return _ffi.string(_lib.OBJ_nid2ln(nid))

    def digest(self, digest_name: str) -> bytes:
        """
        Return the digest of the X509 object.

        :param digest_name: The name of the digest algorithm to use.
        :type digest_name: :py:class:`str`

        :return: The digest of the object, formatted as
            :py:const:`b":"`-delimited hex pairs.
        :rtype: :py:class:`bytes`
        """
        digest = _lib.EVP_get_digestbyname(_byte_string(digest_name))
        if digest == _ffi.NULL:
            raise ValueError('No such digest method')
        result_buffer = _ffi.new('unsigned char[]', _lib.EVP_MAX_MD_SIZE)
        result_length = _ffi.new('unsigned int[]', 1)
        result_length[0] = len(result_buffer)
        digest_result = _lib.X509_digest(self._x509, digest, result_buffer, result_length)
        _openssl_assert(digest_result == 1)
        return b':'.join([b16encode(ch).upper() for ch in _ffi.buffer(result_buffer, result_length[0])])

    def subject_name_hash(self) -> bytes:
        """
        Return the hash of the X509 subject.

        :return: The hash of the subject.
        :rtype: :py:class:`bytes`
        """
        return _lib.X509_subject_name_hash(self._x509)

    def set_serial_number(self, serial: int) -> None:
        """
        Set the serial number of the certificate.

        :param serial: The new serial number.
        :type serial: :py:class:`int`

        :return: :py:data`None`
        """
        if not isinstance(serial, int):
            raise TypeError('serial must be an integer')
        hex_serial = hex(serial)[2:]
        hex_serial_bytes = hex_serial.encode('ascii')
        bignum_serial = _ffi.new('BIGNUM**')
        small_serial = _lib.BN_hex2bn(bignum_serial, hex_serial_bytes)
        if bignum_serial[0] == _ffi.NULL:
            set_result = _lib.ASN1_INTEGER_set(_lib.X509_get_serialNumber(self._x509), small_serial)
            if set_result:
                _raise_current_error()
        else:
            asn1_serial = _lib.BN_to_ASN1_INTEGER(bignum_serial[0], _ffi.NULL)
            _lib.BN_free(bignum_serial[0])
            if asn1_serial == _ffi.NULL:
                _raise_current_error()
            asn1_serial = _ffi.gc(asn1_serial, _lib.ASN1_INTEGER_free)
            set_result = _lib.X509_set_serialNumber(self._x509, asn1_serial)
            _openssl_assert(set_result == 1)

    def get_serial_number(self) -> int:
        """
        Return the serial number of this certificate.

        :return: The serial number.
        :rtype: int
        """
        asn1_serial = _lib.X509_get_serialNumber(self._x509)
        bignum_serial = _lib.ASN1_INTEGER_to_BN(asn1_serial, _ffi.NULL)
        try:
            hex_serial = _lib.BN_bn2hex(bignum_serial)
            try:
                hexstring_serial = _ffi.string(hex_serial)
                serial = int(hexstring_serial, 16)
                return serial
            finally:
                _lib.OPENSSL_free(hex_serial)
        finally:
            _lib.BN_free(bignum_serial)

    def gmtime_adj_notAfter(self, amount: int) -> None:
        """
        Adjust the time stamp on which the certificate stops being valid.

        :param int amount: The number of seconds by which to adjust the
            timestamp.
        :return: ``None``
        """
        if not isinstance(amount, int):
            raise TypeError('amount must be an integer')
        notAfter = _lib.X509_getm_notAfter(self._x509)
        _lib.X509_gmtime_adj(notAfter, amount)

    def gmtime_adj_notBefore(self, amount: int) -> None:
        """
        Adjust the timestamp on which the certificate starts being valid.

        :param amount: The number of seconds by which to adjust the timestamp.
        :return: ``None``
        """
        if not isinstance(amount, int):
            raise TypeError('amount must be an integer')
        notBefore = _lib.X509_getm_notBefore(self._x509)
        _lib.X509_gmtime_adj(notBefore, amount)

    def has_expired(self) -> bool:
        """
        Check whether the certificate has expired.

        :return: ``True`` if the certificate has expired, ``False`` otherwise.
        :rtype: bool
        """
        time_bytes = self.get_notAfter()
        if time_bytes is None:
            raise ValueError('Unable to determine notAfter')
        time_string = time_bytes.decode('utf-8')
        not_after = datetime.datetime.strptime(time_string, '%Y%m%d%H%M%SZ')
        return not_after < datetime.datetime.utcnow()

    def _get_boundary_time(self, which: Any) -> Optional[bytes]:
        return _get_asn1_time(which(self._x509))

    def get_notBefore(self) -> Optional[bytes]:
        """
        Get the timestamp at which the certificate starts being valid.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        :return: A timestamp string, or ``None`` if there is none.
        :rtype: bytes or NoneType
        """
        return self._get_boundary_time(_lib.X509_getm_notBefore)

    def _set_boundary_time(self, which: Callable[..., Any], when: bytes) -> None:
        return _set_asn1_time(which(self._x509), when)

    def set_notBefore(self, when: bytes) -> None:
        """
        Set the timestamp at which the certificate starts being valid.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        :param bytes when: A timestamp string.
        :return: ``None``
        """
        return self._set_boundary_time(_lib.X509_getm_notBefore, when)

    def get_notAfter(self) -> Optional[bytes]:
        """
        Get the timestamp at which the certificate stops being valid.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        :return: A timestamp string, or ``None`` if there is none.
        :rtype: bytes or NoneType
        """
        return self._get_boundary_time(_lib.X509_getm_notAfter)

    def set_notAfter(self, when: bytes) -> None:
        """
        Set the timestamp at which the certificate stops being valid.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        :param bytes when: A timestamp string.
        :return: ``None``
        """
        return self._set_boundary_time(_lib.X509_getm_notAfter, when)

    def _get_name(self, which: Any) -> X509Name:
        name = X509Name.__new__(X509Name)
        name._name = which(self._x509)
        _openssl_assert(name._name != _ffi.NULL)
        name._owner = self
        return name

    def _set_name(self, which: Any, name: X509Name) -> None:
        if not isinstance(name, X509Name):
            raise TypeError('name must be an X509Name')
        set_result = which(self._x509, name._name)
        _openssl_assert(set_result == 1)

    def get_issuer(self) -> X509Name:
        """
        Return the issuer of this certificate.

        This creates a new :class:`X509Name` that wraps the underlying issuer
        name field on the certificate. Modifying it will modify the underlying
        certificate, and will have the effect of modifying any other
        :class:`X509Name` that refers to this issuer.

        :return: The issuer of this certificate.
        :rtype: :class:`X509Name`
        """
        name = self._get_name(_lib.X509_get_issuer_name)
        self._issuer_invalidator.add(name)
        return name

    def set_issuer(self, issuer: X509Name) -> None:
        """
        Set the issuer of this certificate.

        :param issuer: The issuer.
        :type issuer: :py:class:`X509Name`

        :return: ``None``
        """
        self._set_name(_lib.X509_set_issuer_name, issuer)
        self._issuer_invalidator.clear()

    def get_subject(self) -> X509Name:
        """
        Return the subject of this certificate.

        This creates a new :class:`X509Name` that wraps the underlying subject
        name field on the certificate. Modifying it will modify the underlying
        certificate, and will have the effect of modifying any other
        :class:`X509Name` that refers to this subject.

        :return: The subject of this certificate.
        :rtype: :class:`X509Name`
        """
        name = self._get_name(_lib.X509_get_subject_name)
        self._subject_invalidator.add(name)
        return name

    def set_subject(self, subject: X509Name) -> None:
        """
        Set the subject of this certificate.

        :param subject: The subject.
        :type subject: :py:class:`X509Name`

        :return: ``None``
        """
        self._set_name(_lib.X509_set_subject_name, subject)
        self._subject_invalidator.clear()

    def get_extension_count(self) -> int:
        """
        Get the number of extensions on this certificate.

        :return: The number of extensions.
        :rtype: :py:class:`int`

        .. versionadded:: 0.12
        """
        return _lib.X509_get_ext_count(self._x509)

    def add_extensions(self, extensions: Iterable[X509Extension]) -> None:
        """
        Add extensions to the certificate.

        :param extensions: The extensions to add.
        :type extensions: An iterable of :py:class:`X509Extension` objects.
        :return: ``None``
        """
        for ext in extensions:
            if not isinstance(ext, X509Extension):
                raise ValueError('One of the elements is not an X509Extension')
            add_result = _lib.X509_add_ext(self._x509, ext._extension, -1)
            if not add_result:
                _raise_current_error()

    def get_extension(self, index: int) -> X509Extension:
        """
        Get a specific extension of the certificate by index.

        Extensions on a certificate are kept in order. The index
        parameter selects which extension will be returned.

        :param int index: The index of the extension to retrieve.
        :return: The extension at the specified index.
        :rtype: :py:class:`X509Extension`
        :raises IndexError: If the extension index was out of bounds.

        .. versionadded:: 0.12
        """
        ext = X509Extension.__new__(X509Extension)
        ext._extension = _lib.X509_get_ext(self._x509, index)
        if ext._extension == _ffi.NULL:
            raise IndexError('extension index out of bounds')
        extension = _lib.X509_EXTENSION_dup(ext._extension)
        ext._extension = _ffi.gc(extension, _lib.X509_EXTENSION_free)
        return ext