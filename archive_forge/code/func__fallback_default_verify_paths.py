import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _fallback_default_verify_paths(self, file_path, dir_path):
    """
        Default verify paths are based on the compiled version of OpenSSL.
        However, when pyca/cryptography is compiled as a manylinux1 wheel
        that compiled location can potentially be wrong. So, like Go, we
        will try a predefined set of paths and attempt to load roots
        from there.

        :return: None
        """
    for cafile in file_path:
        if os.path.isfile(cafile):
            self.load_verify_locations(cafile)
            break
    for capath in dir_path:
        if os.path.isdir(capath):
            self.load_verify_locations(None, capath)
            break