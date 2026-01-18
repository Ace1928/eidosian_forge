from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class OleFileError(IOError):
    """
    Generic base error for this module.
    """
    pass