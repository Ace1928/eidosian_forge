from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
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