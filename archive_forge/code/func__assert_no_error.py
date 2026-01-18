import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _assert_no_error(error, exception_class=None):
    """
    Checks the return code and throws an exception if there is an error to
    report
    """
    if error == 0:
        return
    cf_error_string = Security.SecCopyErrorMessageString(error, None)
    output = _cf_string_to_unicode(cf_error_string)
    CoreFoundation.CFRelease(cf_error_string)
    if output is None or output == u'':
        output = u'OSStatus %s' % error
    if exception_class is None:
        exception_class = ssl.SSLError
    raise exception_class(output)