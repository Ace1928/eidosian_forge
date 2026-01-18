import os
import platform
import subprocess
import errno
import time
import sys
import unittest
import tempfile
def PeekNamedPipe(handle, desired_bytes):
    c_avail = DWORD()
    c_message = DWORD()
    if desired_bytes > 0:
        c_read = DWORD()
        buffer = ctypes.create_string_buffer(desired_bytes + 1)
        success = ctypes.windll.kernel32.PeekNamedPipe(handle, buffer, desired_bytes, ctypes.byref(c_read), ctypes.byref(c_avail), ctypes.byref(c_message))
        buffer[c_read.value] = null_byte
        return (decode(buffer.value), c_avail.value, c_message.value)
    else:
        success = ctypes.windll.kernel32.PeekNamedPipe(handle, None, desired_bytes, None, ctypes.byref(c_avail), ctypes.byref(c_message))
        return ('', c_avail.value, c_message.value)