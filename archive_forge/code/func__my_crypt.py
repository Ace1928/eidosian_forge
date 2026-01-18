from .err import OperationalError
from functools import partial
import hashlib
def _my_crypt(message1, message2):
    result = bytearray(message1)
    for i in range(len(result)):
        result[i] ^= message2[i]
    return bytes(result)