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
def add_crl(self, crl: 'CRL') -> None:
    """
        Add a certificate revocation list to this store.

        The certificate revocation lists added to a store will only be used if
        the associated flags are configured to check certificate revocation
        lists.

        .. versionadded:: 16.1.0

        :param CRL crl: The certificate revocation list to add to this store.
        :return: ``None`` if the certificate revocation list was added
            successfully.
        """
    _openssl_assert(_lib.X509_STORE_add_crl(self._store, crl._crl) != 0)