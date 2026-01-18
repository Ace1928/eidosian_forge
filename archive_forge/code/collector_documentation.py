import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings

        Return the source of the current line of this frame.  You
        probably want to .strip() it as well, as it is likely to have
        leading whitespace.

        If context is given, then that many lines on either side will
        also be returned.  E.g., context=1 will give 3 lines.
        