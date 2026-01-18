import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
class CollectedException(Bunch):
    """
    This is the result of collection the exception; it contains copies
    of data of interest.
    """
    frames = []
    exception_formatted = None
    exception_type = None
    exception_value = None
    identification_code = None
    date = None
    extra_data = {}