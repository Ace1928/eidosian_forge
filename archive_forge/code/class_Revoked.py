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
class Revoked:
    """
    A certificate revocation.
    """
    _crl_reasons = [b'unspecified', b'keyCompromise', b'CACompromise', b'affiliationChanged', b'superseded', b'cessationOfOperation', b'certificateHold']

    def __init__(self) -> None:
        revoked = _lib.X509_REVOKED_new()
        self._revoked = _ffi.gc(revoked, _lib.X509_REVOKED_free)

    def set_serial(self, hex_str: bytes) -> None:
        """
        Set the serial number.

        The serial number is formatted as a hexadecimal number encoded in
        ASCII.

        :param bytes hex_str: The new serial number.

        :return: ``None``
        """
        bignum_serial = _ffi.gc(_lib.BN_new(), _lib.BN_free)
        bignum_ptr = _ffi.new('BIGNUM**')
        bignum_ptr[0] = bignum_serial
        bn_result = _lib.BN_hex2bn(bignum_ptr, hex_str)
        if not bn_result:
            raise ValueError('bad hex string')
        asn1_serial = _ffi.gc(_lib.BN_to_ASN1_INTEGER(bignum_serial, _ffi.NULL), _lib.ASN1_INTEGER_free)
        _lib.X509_REVOKED_set_serialNumber(self._revoked, asn1_serial)

    def get_serial(self) -> bytes:
        """
        Get the serial number.

        The serial number is formatted as a hexadecimal number encoded in
        ASCII.

        :return: The serial number.
        :rtype: bytes
        """
        bio = _new_mem_buf()
        asn1_int = _lib.X509_REVOKED_get0_serialNumber(self._revoked)
        _openssl_assert(asn1_int != _ffi.NULL)
        result = _lib.i2a_ASN1_INTEGER(bio, asn1_int)
        _openssl_assert(result >= 0)
        return _bio_to_string(bio)

    def _delete_reason(self) -> None:
        for i in range(_lib.X509_REVOKED_get_ext_count(self._revoked)):
            ext = _lib.X509_REVOKED_get_ext(self._revoked, i)
            obj = _lib.X509_EXTENSION_get_object(ext)
            if _lib.OBJ_obj2nid(obj) == _lib.NID_crl_reason:
                _lib.X509_EXTENSION_free(ext)
                _lib.X509_REVOKED_delete_ext(self._revoked, i)
                break

    def set_reason(self, reason: Optional[bytes]) -> None:
        """
        Set the reason of this revocation.

        If :data:`reason` is ``None``, delete the reason instead.

        :param reason: The reason string.
        :type reason: :class:`bytes` or :class:`NoneType`

        :return: ``None``

        .. seealso::

            :meth:`all_reasons`, which gives you a list of all supported
            reasons which you might pass to this method.
        """
        if reason is None:
            self._delete_reason()
        elif not isinstance(reason, bytes):
            raise TypeError('reason must be None or a byte string')
        else:
            reason = reason.lower().replace(b' ', b'')
            reason_code = [r.lower() for r in self._crl_reasons].index(reason)
            new_reason_ext = _lib.ASN1_ENUMERATED_new()
            _openssl_assert(new_reason_ext != _ffi.NULL)
            new_reason_ext = _ffi.gc(new_reason_ext, _lib.ASN1_ENUMERATED_free)
            set_result = _lib.ASN1_ENUMERATED_set(new_reason_ext, reason_code)
            _openssl_assert(set_result != _ffi.NULL)
            self._delete_reason()
            add_result = _lib.X509_REVOKED_add1_ext_i2d(self._revoked, _lib.NID_crl_reason, new_reason_ext, 0, 0)
            _openssl_assert(add_result == 1)

    def get_reason(self) -> Optional[bytes]:
        """
        Get the reason of this revocation.

        :return: The reason, or ``None`` if there is none.
        :rtype: bytes or NoneType

        .. seealso::

            :meth:`all_reasons`, which gives you a list of all supported
            reasons this method might return.
        """
        for i in range(_lib.X509_REVOKED_get_ext_count(self._revoked)):
            ext = _lib.X509_REVOKED_get_ext(self._revoked, i)
            obj = _lib.X509_EXTENSION_get_object(ext)
            if _lib.OBJ_obj2nid(obj) == _lib.NID_crl_reason:
                bio = _new_mem_buf()
                print_result = _lib.X509V3_EXT_print(bio, ext, 0, 0)
                if not print_result:
                    print_result = _lib.M_ASN1_OCTET_STRING_print(bio, _lib.X509_EXTENSION_get_data(ext))
                    _openssl_assert(print_result != 0)
                return _bio_to_string(bio)
        return None

    def all_reasons(self) -> List[bytes]:
        """
        Return a list of all the supported reason strings.

        This list is a copy; modifying it does not change the supported reason
        strings.

        :return: A list of reason strings.
        :rtype: :class:`list` of :class:`bytes`
        """
        return self._crl_reasons[:]

    def set_rev_date(self, when: bytes) -> None:
        """
        Set the revocation timestamp.

        :param bytes when: The timestamp of the revocation,
            as ASN.1 TIME.
        :return: ``None``
        """
        revocationDate = _new_asn1_time(when)
        ret = _lib.X509_REVOKED_set_revocationDate(self._revoked, revocationDate)
        _openssl_assert(ret == 1)

    def get_rev_date(self) -> Optional[bytes]:
        """
        Get the revocation timestamp.

        :return: The timestamp of the revocation, as ASN.1 TIME.
        :rtype: bytes
        """
        dt = _lib.X509_REVOKED_get0_revocationDate(self._revoked)
        return _get_asn1_time(dt)