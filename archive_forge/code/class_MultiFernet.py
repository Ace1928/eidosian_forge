from __future__ import annotations
import base64
import binascii
import os
import time
import typing
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.hmac import HMAC
class MultiFernet:

    def __init__(self, fernets: typing.Iterable[Fernet]):
        fernets = list(fernets)
        if not fernets:
            raise ValueError('MultiFernet requires at least one Fernet instance')
        self._fernets = fernets

    def encrypt(self, msg: bytes) -> bytes:
        return self.encrypt_at_time(msg, int(time.time()))

    def encrypt_at_time(self, msg: bytes, current_time: int) -> bytes:
        return self._fernets[0].encrypt_at_time(msg, current_time)

    def rotate(self, msg: typing.Union[bytes, str]) -> bytes:
        timestamp, data = Fernet._get_unverified_token_data(msg)
        for f in self._fernets:
            try:
                p = f._decrypt_data(data, timestamp, None)
                break
            except InvalidToken:
                pass
        else:
            raise InvalidToken
        iv = os.urandom(16)
        return self._fernets[0]._encrypt_from_parts(p, timestamp, iv)

    def decrypt(self, msg: typing.Union[bytes, str], ttl: typing.Optional[int]=None) -> bytes:
        for f in self._fernets:
            try:
                return f.decrypt(msg, ttl)
            except InvalidToken:
                pass
        raise InvalidToken

    def decrypt_at_time(self, msg: typing.Union[bytes, str], ttl: int, current_time: int) -> bytes:
        for f in self._fernets:
            try:
                return f.decrypt_at_time(msg, ttl, current_time)
            except InvalidToken:
                pass
        raise InvalidToken