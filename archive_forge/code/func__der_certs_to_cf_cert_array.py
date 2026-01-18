import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _der_certs_to_cf_cert_array(certs: list[bytes]) -> CFMutableArrayRef:
    """Builds a CFArray of SecCertificateRefs from a list of DER-encoded certificates.
    Responsibility of the caller to call CoreFoundation.CFRelease on the CFArray.
    """
    cf_array = CoreFoundation.CFArrayCreateMutable(CoreFoundation.kCFAllocatorDefault, 0, ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks))
    if not cf_array:
        raise MemoryError('Unable to allocate memory!')
    for cert_data in certs:
        cf_data = None
        sec_cert_ref = None
        try:
            cf_data = _bytes_to_cf_data_ref(cert_data)
            sec_cert_ref = Security.SecCertificateCreateWithData(CoreFoundation.kCFAllocatorDefault, cf_data)
            CoreFoundation.CFArrayAppendValue(cf_array, sec_cert_ref)
        finally:
            if cf_data:
                CoreFoundation.CFRelease(cf_data)
            if sec_cert_ref:
                CoreFoundation.CFRelease(sec_cert_ref)
    return cf_array