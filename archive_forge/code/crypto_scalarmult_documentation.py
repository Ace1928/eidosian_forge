from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Computes and returns the scalar product of an integer ``n``
    and the given group element on the edwards25519 curve. The integer
    ``n`` is not clamped.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :param p: a :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    