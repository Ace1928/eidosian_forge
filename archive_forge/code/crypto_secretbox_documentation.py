from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Decrypt and returns the encrypted message ``ciphertext`` with the secret
    ``key`` and the nonce ``nonce``.

    :param ciphertext: bytes
    :param nonce: bytes
    :param key: bytes
    :rtype: bytes
    