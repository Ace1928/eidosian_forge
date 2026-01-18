import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _load_items_from_file(keychain, path):
    """
    Given a single file, loads all the trust objects from it into arrays and
    the keychain.
    Returns a tuple of lists: the first list is a list of identities, the
    second a list of certs.
    """
    certificates = []
    identities = []
    result_array = None
    with open(path, 'rb') as f:
        raw_filedata = f.read()
    try:
        filedata = CoreFoundation.CFDataCreate(CoreFoundation.kCFAllocatorDefault, raw_filedata, len(raw_filedata))
        result_array = CoreFoundation.CFArrayRef()
        result = Security.SecItemImport(filedata, None, None, None, 0, None, keychain, ctypes.byref(result_array))
        _assert_no_error(result)
        result_count = CoreFoundation.CFArrayGetCount(result_array)
        for index in range(result_count):
            item = CoreFoundation.CFArrayGetValueAtIndex(result_array, index)
            item = ctypes.cast(item, CoreFoundation.CFTypeRef)
            if _is_cert(item):
                CoreFoundation.CFRetain(item)
                certificates.append(item)
            elif _is_identity(item):
                CoreFoundation.CFRetain(item)
                identities.append(item)
    finally:
        if result_array:
            CoreFoundation.CFRelease(result_array)
        CoreFoundation.CFRelease(filedata)
    return (identities, certificates)