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