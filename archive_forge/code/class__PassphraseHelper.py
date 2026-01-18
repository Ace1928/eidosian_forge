import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
class _PassphraseHelper:

    def __init__(self, type: int, passphrase: Optional[PassphraseCallableT], more_args: bool=False, truncate: bool=False) -> None:
        if type != FILETYPE_PEM and passphrase is not None:
            raise ValueError('only FILETYPE_PEM key format supports encryption')
        self._passphrase = passphrase
        self._more_args = more_args
        self._truncate = truncate
        self._problems: List[Exception] = []

    @property
    def callback(self) -> Any:
        if self._passphrase is None:
            return _ffi.NULL
        elif isinstance(self._passphrase, bytes) or callable(self._passphrase):
            return _ffi.callback('pem_password_cb', self._read_passphrase)
        else:
            raise TypeError('Last argument must be a byte string or a callable.')

    @property
    def callback_args(self) -> Any:
        if self._passphrase is None:
            return _ffi.NULL
        elif isinstance(self._passphrase, bytes) or callable(self._passphrase):
            return _ffi.NULL
        else:
            raise TypeError('Last argument must be a byte string or a callable.')

    def raise_if_problem(self, exceptionType: Type[Exception]=Error) -> None:
        if self._problems:
            try:
                _exception_from_error_queue(exceptionType)
            except exceptionType:
                pass
            raise self._problems.pop(0)

    def _read_passphrase(self, buf: Any, size: int, rwflag: Any, userdata: Any) -> int:
        try:
            if callable(self._passphrase):
                if self._more_args:
                    result = self._passphrase(size, rwflag, userdata)
                else:
                    result = self._passphrase(rwflag)
            else:
                assert self._passphrase is not None
                result = self._passphrase
            if not isinstance(result, bytes):
                raise ValueError('Bytes expected')
            if len(result) > size:
                if self._truncate:
                    result = result[:size]
                else:
                    raise ValueError('passphrase returned by callback is too long')
            for i in range(len(result)):
                buf[i] = result[i:i + 1]
            return len(result)
        except Exception as e:
            self._problems.append(e)
            return 0