from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import ctypes
from ctypes import windll  # pylint: disable=g-importing-member
from ctypes import wintypes  # pylint: disable=g-importing-member
class WindowsException(Exception):

    def __init__(self, extra_data=None):
        windows_error_code = get_last_error()
        message = 'Windows Error: 0x%0x' % windows_error_code
        if extra_data:
            message += '\nExtra Info: %s' % extra_data
        super(WindowsException, self).__init__(message)