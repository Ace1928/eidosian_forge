import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _load_client_cert_chain(keychain, *paths):
    """
    Load certificates and maybe keys from a number of files. Has the end goal
    of returning a CFArray containing one SecIdentityRef, and then zero or more
    SecCertificateRef objects, suitable for use as a client certificate trust
    chain.
    """
    certificates = []
    identities = []
    paths = (path for path in paths if path)
    try:
        for file_path in paths:
            new_identities, new_certs = _load_items_from_file(keychain, file_path)
            identities.extend(new_identities)
            certificates.extend(new_certs)
        if not identities:
            new_identity = Security.SecIdentityRef()
            status = Security.SecIdentityCreateWithCertificate(keychain, certificates[0], ctypes.byref(new_identity))
            _assert_no_error(status)
            identities.append(new_identity)
            CoreFoundation.CFRelease(certificates.pop(0))
        trust_chain = CoreFoundation.CFArrayCreateMutable(CoreFoundation.kCFAllocatorDefault, 0, ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks))
        for item in itertools.chain(identities, certificates):
            CoreFoundation.CFArrayAppendValue(trust_chain, item)
        return trust_chain
    finally:
        for obj in itertools.chain(identities, certificates):
            CoreFoundation.CFRelease(obj)