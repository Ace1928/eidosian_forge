from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
class OCSPRequestBuilder:

    def __init__(self, request: typing.Optional[typing.Tuple[x509.Certificate, x509.Certificate, hashes.HashAlgorithm]]=None, request_hash: typing.Optional[typing.Tuple[bytes, bytes, int, hashes.HashAlgorithm]]=None, extensions: typing.List[x509.Extension[x509.ExtensionType]]=[]) -> None:
        self._request = request
        self._request_hash = request_hash
        self._extensions = extensions

    def add_certificate(self, cert: x509.Certificate, issuer: x509.Certificate, algorithm: hashes.HashAlgorithm) -> OCSPRequestBuilder:
        if self._request is not None or self._request_hash is not None:
            raise ValueError('Only one certificate can be added to a request')
        _verify_algorithm(algorithm)
        if not isinstance(cert, x509.Certificate) or not isinstance(issuer, x509.Certificate):
            raise TypeError('cert and issuer must be a Certificate')
        return OCSPRequestBuilder((cert, issuer, algorithm), self._request_hash, self._extensions)

    def add_certificate_by_hash(self, issuer_name_hash: bytes, issuer_key_hash: bytes, serial_number: int, algorithm: hashes.HashAlgorithm) -> OCSPRequestBuilder:
        if self._request is not None or self._request_hash is not None:
            raise ValueError('Only one certificate can be added to a request')
        if not isinstance(serial_number, int):
            raise TypeError('serial_number must be an integer')
        _verify_algorithm(algorithm)
        utils._check_bytes('issuer_name_hash', issuer_name_hash)
        utils._check_bytes('issuer_key_hash', issuer_key_hash)
        if algorithm.digest_size != len(issuer_name_hash) or algorithm.digest_size != len(issuer_key_hash):
            raise ValueError('issuer_name_hash and issuer_key_hash must be the same length as the digest size of the algorithm')
        return OCSPRequestBuilder(self._request, (issuer_name_hash, issuer_key_hash, serial_number, algorithm), self._extensions)

    def add_extension(self, extval: x509.ExtensionType, critical: bool) -> OCSPRequestBuilder:
        if not isinstance(extval, x509.ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = x509.Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return OCSPRequestBuilder(self._request, self._request_hash, self._extensions + [extension])

    def build(self) -> OCSPRequest:
        if self._request is None and self._request_hash is None:
            raise ValueError('You must add a certificate before building')
        return ocsp.create_ocsp_request(self)