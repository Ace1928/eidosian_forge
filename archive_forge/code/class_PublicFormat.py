from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class PublicFormat(utils.Enum):
    SubjectPublicKeyInfo = 'X.509 subjectPublicKeyInfo with PKCS#1'
    PKCS1 = 'Raw PKCS#1'
    OpenSSH = 'OpenSSH'
    Raw = 'Raw'
    CompressedPoint = 'X9.62 Compressed Point'
    UncompressedPoint = 'X9.62 Uncompressed Point'