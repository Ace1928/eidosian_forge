from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Decrypts and returns an encrypted message ``ciphertext``, using the
    recipent's secret key ``sk`` and the sender's ephemeral public key
    embedded in the sealed box. The box contruct nonce is derived from
    the recipient's public key ``pk`` and the sender's public key.

    :param ciphertext: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes

    .. versionadded:: 1.2
    