from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Reduce integer ``s`` to ``s`` modulo ``L``, where ``L`` is the order
    of the main subgroup.

    :param s: a :py:data:`.crypto_core_ed25519_NONREDUCEDSCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    