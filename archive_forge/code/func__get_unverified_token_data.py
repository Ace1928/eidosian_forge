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
@staticmethod
def _get_unverified_token_data(token: typing.Union[bytes, str]) -> typing.Tuple[int, bytes]:
    if not isinstance(token, (str, bytes)):
        raise TypeError('token must be bytes or str')
    try:
        data = base64.urlsafe_b64decode(token)
    except (TypeError, binascii.Error):
        raise InvalidToken
    if not data or data[0] != 128:
        raise InvalidToken
    if len(data) < 9:
        raise InvalidToken
    timestamp = int.from_bytes(data[1:9], byteorder='big')
    return (timestamp, data)