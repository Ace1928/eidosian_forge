from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA3_256(HashAlgorithm):
    name = 'sha3-256'
    digest_size = 32
    block_size = None