from __future__ import absolute_import
import io
import time
class PortNotOpenError(SerialException):
    """Port is not open"""

    def __init__(self):
        super(PortNotOpenError, self).__init__('Attempting to use a port that is not open')