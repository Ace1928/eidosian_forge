import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
class ExceptionFrame(Bunch):
    """
    This represents one frame of the exception.  Each frame is a
    context in the call stack, typically represented by a line
    number and module name in the traceback.
    """
    modname = None
    filename = None
    lineno = None
    revision = None
    name = None
    supplement = None
    supplement_exception = None
    traceback_info = None
    traceback_hide = False
    traceback_decorator = None
    tbid = None

    def get_source_line(self, context=0):
        """
        Return the source of the current line of this frame.  You
        probably want to .strip() it as well, as it is likely to have
        leading whitespace.

        If context is given, then that many lines on either side will
        also be returned.  E.g., context=1 will give 3 lines.
        """
        if not self.filename or not self.lineno:
            return None
        lines = []
        for lineno in range(self.lineno - context, self.lineno + context + 1):
            lines.append(linecache.getline(self.filename, lineno))
        return ''.join(lines)