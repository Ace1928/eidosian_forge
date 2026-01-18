import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _raise_passphrase_exception(self):
    if self._passphrase_helper is not None:
        self._passphrase_helper.raise_if_problem(Error)
    _raise_current_error()