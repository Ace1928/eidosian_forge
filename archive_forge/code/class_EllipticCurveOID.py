from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurveOID:
    SECP192R1 = ObjectIdentifier('1.2.840.10045.3.1.1')
    SECP224R1 = ObjectIdentifier('1.3.132.0.33')
    SECP256K1 = ObjectIdentifier('1.3.132.0.10')
    SECP256R1 = ObjectIdentifier('1.2.840.10045.3.1.7')
    SECP384R1 = ObjectIdentifier('1.3.132.0.34')
    SECP521R1 = ObjectIdentifier('1.3.132.0.35')
    BRAINPOOLP256R1 = ObjectIdentifier('1.3.36.3.3.2.8.1.1.7')
    BRAINPOOLP384R1 = ObjectIdentifier('1.3.36.3.3.2.8.1.1.11')
    BRAINPOOLP512R1 = ObjectIdentifier('1.3.36.3.3.2.8.1.1.13')
    SECT163K1 = ObjectIdentifier('1.3.132.0.1')
    SECT163R2 = ObjectIdentifier('1.3.132.0.15')
    SECT233K1 = ObjectIdentifier('1.3.132.0.26')
    SECT233R1 = ObjectIdentifier('1.3.132.0.27')
    SECT283K1 = ObjectIdentifier('1.3.132.0.16')
    SECT283R1 = ObjectIdentifier('1.3.132.0.17')
    SECT409K1 = ObjectIdentifier('1.3.132.0.36')
    SECT409R1 = ObjectIdentifier('1.3.132.0.37')
    SECT571K1 = ObjectIdentifier('1.3.132.0.38')
    SECT571R1 = ObjectIdentifier('1.3.132.0.39')