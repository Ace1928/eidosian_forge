from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat.primitives.hashes import HashAlgorithm
class BestAvailableEncryption(KeySerializationEncryption):

    def __init__(self, password: bytes):
        if not isinstance(password, bytes) or len(password) == 0:
            raise ValueError('Password must be 1 or more bytes.')
        self.password = password