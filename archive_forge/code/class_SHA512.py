from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA512(HashAlgorithm):
    name = 'sha512'
    digest_size = 64
    block_size = 128