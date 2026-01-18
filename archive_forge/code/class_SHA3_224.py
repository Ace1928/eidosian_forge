from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA3_224(HashAlgorithm):
    name = 'sha3-224'
    digest_size = 28
    block_size = None