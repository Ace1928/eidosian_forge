from __future__ import absolute_import
import ctypes
import time
from serial import win32
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, PortNotOpenError, SerialTimeoutException
def _cancel_overlapped_io(self, overlapped):
    """Cancel a blocking read operation, may be called from other thread"""
    rc = win32.DWORD()
    err = win32.GetOverlappedResult(self._port_handle, ctypes.byref(overlapped), ctypes.byref(rc), False)
    if not err and win32.GetLastError() in (win32.ERROR_IO_PENDING, win32.ERROR_IO_INCOMPLETE):
        win32.CancelIoEx(self._port_handle, overlapped)