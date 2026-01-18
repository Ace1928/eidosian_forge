import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
class MacError(DecodeError):

    def __init__(self, code):
        if code == ERROR_TYPE:
            msg = 'unsupported audio type'
        elif code == ERROR_FORMAT:
            msg = 'unsupported format'
        else:
            msg = 'error %i' % code
        super().__init__(msg)