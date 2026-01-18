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
def add_extensions(self, extensions: Iterable[X509Extension]) -> None:
    """
        Add extensions to the certificate.

        :param extensions: The extensions to add.
        :type extensions: An iterable of :py:class:`X509Extension` objects.
        :return: ``None``
        """
    for ext in extensions:
        if not isinstance(ext, X509Extension):
            raise ValueError('One of the elements is not an X509Extension')
        add_result = _lib.X509_add_ext(self._x509, ext._extension, -1)
        if not add_result:
            _raise_current_error()