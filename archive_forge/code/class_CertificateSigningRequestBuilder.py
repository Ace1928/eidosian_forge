from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
class CertificateSigningRequestBuilder:

    def __init__(self, subject_name: typing.Optional[Name]=None, extensions: typing.List[Extension[ExtensionType]]=[], attributes: typing.List[typing.Tuple[ObjectIdentifier, bytes, typing.Optional[int]]]=[]):
        """
        Creates an empty X.509 certificate request (v1).
        """
        self._subject_name = subject_name
        self._extensions = extensions
        self._attributes = attributes

    def subject_name(self, name: Name) -> CertificateSigningRequestBuilder:
        """
        Sets the certificate requestor's distinguished name.
        """
        if not isinstance(name, Name):
            raise TypeError('Expecting x509.Name object.')
        if self._subject_name is not None:
            raise ValueError('The subject name may only be set once.')
        return CertificateSigningRequestBuilder(name, self._extensions, self._attributes)

    def add_extension(self, extval: ExtensionType, critical: bool) -> CertificateSigningRequestBuilder:
        """
        Adds an X.509 extension to the certificate request.
        """
        if not isinstance(extval, ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return CertificateSigningRequestBuilder(self._subject_name, self._extensions + [extension], self._attributes)

    def add_attribute(self, oid: ObjectIdentifier, value: bytes, *, _tag: typing.Optional[_ASN1Type]=None) -> CertificateSigningRequestBuilder:
        """
        Adds an X.509 attribute with an OID and associated value.
        """
        if not isinstance(oid, ObjectIdentifier):
            raise TypeError('oid must be an ObjectIdentifier')
        if not isinstance(value, bytes):
            raise TypeError('value must be bytes')
        if _tag is not None and (not isinstance(_tag, _ASN1Type)):
            raise TypeError('tag must be _ASN1Type')
        _reject_duplicate_attribute(oid, self._attributes)
        if _tag is not None:
            tag = _tag.value
        else:
            tag = None
        return CertificateSigningRequestBuilder(self._subject_name, self._extensions, self._attributes + [(oid, value, tag)])

    def sign(self, private_key: CertificateIssuerPrivateKeyTypes, algorithm: typing.Optional[_AllowedHashTypes], backend: typing.Any=None) -> CertificateSigningRequest:
        """
        Signs the request using the requestor's private key.
        """
        if self._subject_name is None:
            raise ValueError('A CertificateSigningRequest must have a subject')
        return rust_x509.create_x509_csr(self, private_key, algorithm)