from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA3_512(HashAlgorithm):
    name = 'sha3-512'
    digest_size = 64
    block_size = None