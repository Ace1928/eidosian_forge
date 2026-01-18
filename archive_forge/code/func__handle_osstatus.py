import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import (
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
def _handle_osstatus(result: OSStatus, _: typing.Any, args: typing.Any) -> typing.Any:
    """
    Raises an error if the OSStatus value is non-zero.
    """
    if int(result) == 0:
        return args
    error_message_cfstring = None
    try:
        error_message_cfstring = Security.SecCopyErrorMessageString(result, None)
        error_message_cfstring_c_void_p = ctypes.cast(error_message_cfstring, ctypes.POINTER(ctypes.c_void_p))
        message = CoreFoundation.CFStringGetCStringPtr(error_message_cfstring_c_void_p, CFConst.kCFStringEncodingUTF8)
        if message is None:
            buffer = ctypes.create_string_buffer(1024)
            result = CoreFoundation.CFStringGetCString(error_message_cfstring_c_void_p, buffer, 1024, CFConst.kCFStringEncodingUTF8)
            if not result:
                raise OSError('Error copying C string from CFStringRef')
            message = buffer.value
    finally:
        if error_message_cfstring is not None:
            CoreFoundation.CFRelease(error_message_cfstring)
    if message is None or message == '':
        message = f'SecureTransport operation returned a non-zero OSStatus: {result}'
    raise ssl.SSLError(message)