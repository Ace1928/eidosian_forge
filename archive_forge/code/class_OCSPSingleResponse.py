from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
class OCSPSingleResponse(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def certificate_status(self) -> OCSPCertStatus:
        """
        The status of the certificate (an element from the OCSPCertStatus enum)
        """

    @property
    @abc.abstractmethod
    def revocation_time(self) -> typing.Optional[datetime.datetime]:
        """
        The date of when the certificate was revoked or None if not
        revoked.
        """

    @property
    @abc.abstractmethod
    def revocation_reason(self) -> typing.Optional[x509.ReasonFlags]:
        """
        The reason the certificate was revoked or None if not specified or
        not revoked.
        """

    @property
    @abc.abstractmethod
    def this_update(self) -> datetime.datetime:
        """
        The most recent time at which the status being indicated is known by
        the responder to have been correct
        """

    @property
    @abc.abstractmethod
    def next_update(self) -> typing.Optional[datetime.datetime]:
        """
        The time when newer information will be available
        """

    @property
    @abc.abstractmethod
    def issuer_key_hash(self) -> bytes:
        """
        The hash of the issuer public key
        """

    @property
    @abc.abstractmethod
    def issuer_name_hash(self) -> bytes:
        """
        The hash of the issuer name
        """

    @property
    @abc.abstractmethod
    def hash_algorithm(self) -> hashes.HashAlgorithm:
        """
        The hash algorithm used in the issuer name and key hashes
        """

    @property
    @abc.abstractmethod
    def serial_number(self) -> int:
        """
        The serial number of the cert whose status is being checked
        """