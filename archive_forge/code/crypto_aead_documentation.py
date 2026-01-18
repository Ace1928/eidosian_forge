from typing import Optional
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Decrypt the given ``ciphertext`` using the long-nonces xchacha20poly1305
    construction.

    :param ciphertext: authenticated ciphertext
    :type ciphertext: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: message
    :rtype: bytes
    