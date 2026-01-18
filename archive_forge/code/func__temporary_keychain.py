import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _temporary_keychain():
    """
    This function creates a temporary Mac keychain that we can use to work with
    credentials. This keychain uses a one-time password and a temporary file to
    store the data. We expect to have one keychain per socket. The returned
    SecKeychainRef must be freed by the caller, including calling
    SecKeychainDelete.

    Returns a tuple of the SecKeychainRef and the path to the temporary
    directory that contains it.
    """
    random_bytes = os.urandom(40)
    filename = base64.b16encode(random_bytes[:8]).decode('utf-8')
    password = base64.b16encode(random_bytes[8:])
    tempdirectory = tempfile.mkdtemp()
    keychain_path = os.path.join(tempdirectory, filename).encode('utf-8')
    keychain = Security.SecKeychainRef()
    status = Security.SecKeychainCreate(keychain_path, len(password), password, False, None, ctypes.byref(keychain))
    _assert_no_error(status)
    return (keychain, tempdirectory)