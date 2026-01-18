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
class CertificateSigningRequest(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Checks equality.
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Computes a hash.
        """

    @abc.abstractmethod
    def public_key(self) -> CertificatePublicKeyTypes:
        """
        Returns the public key
        """

    @property
    @abc.abstractmethod
    def subject(self) -> Name:
        """
        Returns the subject name object.
        """

    @property
    @abc.abstractmethod
    def signature_hash_algorithm(self) -> typing.Optional[hashes.HashAlgorithm]:
        """
        Returns a HashAlgorithm corresponding to the type of the digest signed
        in the certificate.
        """

    @property
    @abc.abstractmethod
    def signature_algorithm_oid(self) -> ObjectIdentifier:
        """
        Returns the ObjectIdentifier of the signature algorithm.
        """

    @property
    @abc.abstractmethod
    def extensions(self) -> Extensions:
        """
        Returns the extensions in the signing request.
        """

    @property
    @abc.abstractmethod
    def attributes(self) -> Attributes:
        """
        Returns an Attributes object.
        """

    @abc.abstractmethod
    def public_bytes(self, encoding: serialization.Encoding) -> bytes:
        """
        Encodes the request to PEM or DER format.
        """

    @property
    @abc.abstractmethod
    def signature(self) -> bytes:
        """
        Returns the signature bytes.
        """

    @property
    @abc.abstractmethod
    def tbs_certrequest_bytes(self) -> bytes:
        """
        Returns the PKCS#10 CertificationRequestInfo bytes as defined in RFC
        2986.
        """

    @property
    @abc.abstractmethod
    def is_signature_valid(self) -> bool:
        """
        Verifies signature of signing request.
        """

    @abc.abstractmethod
    def get_attribute_for_oid(self, oid: ObjectIdentifier) -> bytes:
        """
        Get the attribute value for a given OID.
        """