from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class NotOleFileError(OleFileError):
    """
    Error raised when the opened file is not an OLE file.
    """
    pass