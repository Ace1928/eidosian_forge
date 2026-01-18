from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA384(HashAlgorithm):
    name = 'sha384'
    digest_size = 48
    block_size = 128