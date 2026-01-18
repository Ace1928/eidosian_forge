from __future__ import annotations
import abc
import datetime
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class SignedCertificateTimestamp(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def version(self) -> Version:
        """
        Returns the SCT version.
        """

    @property
    @abc.abstractmethod
    def log_id(self) -> bytes:
        """
        Returns an identifier indicating which log this SCT is for.
        """

    @property
    @abc.abstractmethod
    def timestamp(self) -> datetime.datetime:
        """
        Returns the timestamp for this SCT.
        """

    @property
    @abc.abstractmethod
    def entry_type(self) -> LogEntryType:
        """
        Returns whether this is an SCT for a certificate or pre-certificate.
        """

    @property
    @abc.abstractmethod
    def signature_hash_algorithm(self) -> HashAlgorithm:
        """
        Returns the hash algorithm used for the SCT's signature.
        """

    @property
    @abc.abstractmethod
    def signature_algorithm(self) -> SignatureAlgorithm:
        """
        Returns the signing algorithm used for the SCT's signature.
        """

    @property
    @abc.abstractmethod
    def signature(self) -> bytes:
        """
        Returns the signature for this SCT.
        """

    @property
    @abc.abstractmethod
    def extension_bytes(self) -> bytes:
        """
        Returns the raw bytes of any extensions for this SCT.
        """