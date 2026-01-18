from .err import OperationalError
from functools import partial
import hashlib
def _xor_password(password, salt):
    salt = salt[:SCRAMBLE_LENGTH]
    password_bytes = bytearray(password)
    salt_len = len(salt)
    for i in range(len(password_bytes)):
        password_bytes[i] ^= salt[i % salt_len]
    return bytes(password_bytes)