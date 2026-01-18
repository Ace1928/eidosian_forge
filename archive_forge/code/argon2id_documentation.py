import nacl.bindings
import nacl.encoding
from . import _argon2

    Hashes a password with a random salt, using the memory-hard
    argon2id construct and returning an ascii string that has all
    the needed info to check against a future password

    The default settings for opslimit and memlimit are those deemed
    correct for the interactive user login case.

    :param bytes password:
    :param int opslimit:
    :param int memlimit:
    :rtype: bytes

    .. versionadded:: 1.2
    