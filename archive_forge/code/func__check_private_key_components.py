from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def _check_private_key_components(p: int, q: int, private_exponent: int, dmp1: int, dmq1: int, iqmp: int, public_exponent: int, modulus: int) -> None:
    if modulus < 3:
        raise ValueError('modulus must be >= 3.')
    if p >= modulus:
        raise ValueError('p must be < modulus.')
    if q >= modulus:
        raise ValueError('q must be < modulus.')
    if dmp1 >= modulus:
        raise ValueError('dmp1 must be < modulus.')
    if dmq1 >= modulus:
        raise ValueError('dmq1 must be < modulus.')
    if iqmp >= modulus:
        raise ValueError('iqmp must be < modulus.')
    if private_exponent >= modulus:
        raise ValueError('private_exponent must be < modulus.')
    if public_exponent < 3 or public_exponent >= modulus:
        raise ValueError('public_exponent must be >= 3 and < modulus.')
    if public_exponent & 1 == 0:
        raise ValueError('public_exponent must be odd.')
    if dmp1 & 1 == 0:
        raise ValueError('dmp1 must be odd.')
    if dmq1 & 1 == 0:
        raise ValueError('dmq1 must be odd.')
    if p * q != modulus:
        raise ValueError('p*q must equal modulus.')