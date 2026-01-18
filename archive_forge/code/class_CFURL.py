import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
class CFURL(CFObject):

    def __init__(self, filename):
        if not isinstance(filename, bytes):
            filename = filename.encode(sys.getfilesystemencoding())
        filename = os.path.abspath(os.path.expanduser(filename))
        url = _corefoundation.CFURLCreateFromFileSystemRepresentation(0, filename, len(filename), False)
        super().__init__(url)

    def __str__(self):
        cfstr = _corefoundation.CFURLGetString(self._obj)
        out = _corefoundation.CFStringGetCStringPtr(cfstr, 0)
        return out