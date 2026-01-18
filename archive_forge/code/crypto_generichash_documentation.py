from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

        Raise the same exception as hashlib's blake implementation
        on copy.copy()
        