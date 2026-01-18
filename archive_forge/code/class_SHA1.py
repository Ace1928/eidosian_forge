from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA1(HashAlgorithm):
    name = 'sha1'
    digest_size = 20
    block_size = 64