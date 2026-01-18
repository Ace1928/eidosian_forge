from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def _checkparams(digest_size: int, key: bytes, salt: bytes, person: bytes) -> None:
    """Check hash parameters"""
    ensure(isinstance(key, bytes), 'Key must be a bytes sequence', raising=exc.TypeError)
    ensure(isinstance(salt, bytes), 'Salt must be a bytes sequence', raising=exc.TypeError)
    ensure(isinstance(person, bytes), 'Person must be a bytes sequence', raising=exc.TypeError)
    ensure(isinstance(digest_size, int), 'Digest size must be an integer number', raising=exc.TypeError)
    ensure(digest_size <= crypto_generichash_BYTES_MAX, _TOOBIG.format('Digest_size', crypto_generichash_BYTES_MAX), raising=exc.ValueError)
    ensure(len(key) <= crypto_generichash_KEYBYTES_MAX, _OVERLONG.format('Key', crypto_generichash_KEYBYTES_MAX), raising=exc.ValueError)
    ensure(len(salt) <= crypto_generichash_SALTBYTES, _OVERLONG.format('Salt', crypto_generichash_SALTBYTES), raising=exc.ValueError)
    ensure(len(person) <= crypto_generichash_PERSONALBYTES, _OVERLONG.format('Person', crypto_generichash_PERSONALBYTES), raising=exc.ValueError)