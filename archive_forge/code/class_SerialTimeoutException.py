from __future__ import absolute_import
import io
import time
class SerialTimeoutException(SerialException):
    """Write timeouts give an exception"""