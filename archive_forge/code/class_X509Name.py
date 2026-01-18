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
@functools.total_ordering
class X509Name:
    """
    An X.509 Distinguished Name.

    :ivar countryName: The country of the entity.
    :ivar C: Alias for  :py:attr:`countryName`.

    :ivar stateOrProvinceName: The state or province of the entity.
    :ivar ST: Alias for :py:attr:`stateOrProvinceName`.

    :ivar localityName: The locality of the entity.
    :ivar L: Alias for :py:attr:`localityName`.

    :ivar organizationName: The organization name of the entity.
    :ivar O: Alias for :py:attr:`organizationName`.

    :ivar organizationalUnitName: The organizational unit of the entity.
    :ivar OU: Alias for :py:attr:`organizationalUnitName`

    :ivar commonName: The common name of the entity.
    :ivar CN: Alias for :py:attr:`commonName`.

    :ivar emailAddress: The e-mail address of the entity.
    """

    def __init__(self, name: 'X509Name') -> None:
        """
        Create a new X509Name, copying the given X509Name instance.

        :param name: The name to copy.
        :type name: :py:class:`X509Name`
        """
        name = _lib.X509_NAME_dup(name._name)
        self._name: Any = _ffi.gc(name, _lib.X509_NAME_free)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            return super(X509Name, self).__setattr__(name, value)
        if type(name) is not str:
            raise TypeError("attribute name must be string, not '%.200s'" % (type(value).__name__,))
        nid = _lib.OBJ_txt2nid(_byte_string(name))
        if nid == _lib.NID_undef:
            try:
                _raise_current_error()
            except Error:
                pass
            raise AttributeError('No such attribute')
        for i in range(_lib.X509_NAME_entry_count(self._name)):
            ent = _lib.X509_NAME_get_entry(self._name, i)
            ent_obj = _lib.X509_NAME_ENTRY_get_object(ent)
            ent_nid = _lib.OBJ_obj2nid(ent_obj)
            if nid == ent_nid:
                ent = _lib.X509_NAME_delete_entry(self._name, i)
                _lib.X509_NAME_ENTRY_free(ent)
                break
        if isinstance(value, str):
            value = value.encode('utf-8')
        add_result = _lib.X509_NAME_add_entry_by_NID(self._name, nid, _lib.MBSTRING_UTF8, value, -1, -1, 0)
        if not add_result:
            _raise_current_error()

    def __getattr__(self, name: str) -> Optional[str]:
        """
        Find attribute. An X509Name object has the following attributes:
        countryName (alias C), stateOrProvince (alias ST), locality (alias L),
        organization (alias O), organizationalUnit (alias OU), commonName
        (alias CN) and more...
        """
        nid = _lib.OBJ_txt2nid(_byte_string(name))
        if nid == _lib.NID_undef:
            try:
                _raise_current_error()
            except Error:
                pass
            raise AttributeError('No such attribute')
        entry_index = _lib.X509_NAME_get_index_by_NID(self._name, nid, -1)
        if entry_index == -1:
            return None
        entry = _lib.X509_NAME_get_entry(self._name, entry_index)
        data = _lib.X509_NAME_ENTRY_get_data(entry)
        result_buffer = _ffi.new('unsigned char**')
        data_length = _lib.ASN1_STRING_to_UTF8(result_buffer, data)
        _openssl_assert(data_length >= 0)
        try:
            result = _ffi.buffer(result_buffer[0], data_length)[:].decode('utf-8')
        finally:
            _lib.OPENSSL_free(result_buffer[0])
        return result

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, X509Name):
            return NotImplemented
        return _lib.X509_NAME_cmp(self._name, other._name) == 0

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, X509Name):
            return NotImplemented
        return _lib.X509_NAME_cmp(self._name, other._name) < 0

    def __repr__(self) -> str:
        """
        String representation of an X509Name
        """
        result_buffer = _ffi.new('char[]', 512)
        format_result = _lib.X509_NAME_oneline(self._name, result_buffer, len(result_buffer))
        _openssl_assert(format_result != _ffi.NULL)
        return "<X509Name object '%s'>" % (_ffi.string(result_buffer).decode('utf-8'),)

    def hash(self) -> int:
        """
        Return an integer representation of the first four bytes of the
        MD5 digest of the DER representation of the name.

        This is the Python equivalent of OpenSSL's ``X509_NAME_hash``.

        :return: The (integer) hash of this name.
        :rtype: :py:class:`int`
        """
        return _lib.X509_NAME_hash(self._name)

    def der(self) -> bytes:
        """
        Return the DER encoding of this name.

        :return: The DER encoded form of this name.
        :rtype: :py:class:`bytes`
        """
        result_buffer = _ffi.new('unsigned char**')
        encode_result = _lib.i2d_X509_NAME(self._name, result_buffer)
        _openssl_assert(encode_result >= 0)
        string_result = _ffi.buffer(result_buffer[0], encode_result)[:]
        _lib.OPENSSL_free(result_buffer[0])
        return string_result

    def get_components(self) -> List[Tuple[bytes, bytes]]:
        """
        Returns the components of this name, as a sequence of 2-tuples.

        :return: The components of this name.
        :rtype: :py:class:`list` of ``name, value`` tuples.
        """
        result = []
        for i in range(_lib.X509_NAME_entry_count(self._name)):
            ent = _lib.X509_NAME_get_entry(self._name, i)
            fname = _lib.X509_NAME_ENTRY_get_object(ent)
            fval = _lib.X509_NAME_ENTRY_get_data(ent)
            nid = _lib.OBJ_obj2nid(fname)
            name = _lib.OBJ_nid2sn(nid)
            value = _ffi.buffer(_lib.ASN1_STRING_get0_data(fval), _lib.ASN1_STRING_length(fval))[:]
            result.append((_ffi.string(name), value))
        return result