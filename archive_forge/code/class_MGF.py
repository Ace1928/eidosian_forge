from __future__ import annotations
import abc
import typing
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
from cryptography.hazmat.primitives.asymmetric import rsa
class MGF(metaclass=abc.ABCMeta):
    _algorithm: hashes.HashAlgorithm