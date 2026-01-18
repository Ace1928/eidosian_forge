import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def _check_argon2_limits_alg(opslimit: int, memlimit: int, alg: int) -> None:
    if alg == crypto_pwhash_ALG_ARGON2I13:
        if memlimit < crypto_pwhash_argon2i_MEMLIMIT_MIN:
            raise exc.ValueError('memlimit must be at least {} bytes'.format(crypto_pwhash_argon2i_MEMLIMIT_MIN))
        elif memlimit > crypto_pwhash_argon2i_MEMLIMIT_MAX:
            raise exc.ValueError('memlimit must be at most {} bytes'.format(crypto_pwhash_argon2i_MEMLIMIT_MAX))
        if opslimit < crypto_pwhash_argon2i_OPSLIMIT_MIN:
            raise exc.ValueError('opslimit must be at least {}'.format(crypto_pwhash_argon2i_OPSLIMIT_MIN))
        elif opslimit > crypto_pwhash_argon2i_OPSLIMIT_MAX:
            raise exc.ValueError('opslimit must be at most {}'.format(crypto_pwhash_argon2i_OPSLIMIT_MAX))
    elif alg == crypto_pwhash_ALG_ARGON2ID13:
        if memlimit < crypto_pwhash_argon2id_MEMLIMIT_MIN:
            raise exc.ValueError('memlimit must be at least {} bytes'.format(crypto_pwhash_argon2id_MEMLIMIT_MIN))
        elif memlimit > crypto_pwhash_argon2id_MEMLIMIT_MAX:
            raise exc.ValueError('memlimit must be at most {} bytes'.format(crypto_pwhash_argon2id_MEMLIMIT_MAX))
        if opslimit < crypto_pwhash_argon2id_OPSLIMIT_MIN:
            raise exc.ValueError('opslimit must be at least {}'.format(crypto_pwhash_argon2id_OPSLIMIT_MIN))
        elif opslimit > crypto_pwhash_argon2id_OPSLIMIT_MAX:
            raise exc.ValueError('opslimit must be at most {}'.format(crypto_pwhash_argon2id_OPSLIMIT_MAX))
    else:
        raise exc.TypeError('Unsupported algorithm')