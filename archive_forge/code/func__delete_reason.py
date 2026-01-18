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
def _delete_reason(self) -> None:
    for i in range(_lib.X509_REVOKED_get_ext_count(self._revoked)):
        ext = _lib.X509_REVOKED_get_ext(self._revoked, i)
        obj = _lib.X509_EXTENSION_get_object(ext)
        if _lib.OBJ_obj2nid(obj) == _lib.NID_crl_reason:
            _lib.X509_EXTENSION_free(ext)
            _lib.X509_REVOKED_delete_ext(self._revoked, i)
            break