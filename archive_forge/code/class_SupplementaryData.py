import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
class SupplementaryData(Bunch):
    """
    The result of __traceback_supplement__.  We don't keep the
    supplement object around, for fear of GC problems and whatnot.
    (@@: Maybe I'm being too superstitious about copying only specific
    information over)
    """
    object = None
    source_url = None
    line = None
    column = None
    expression = None
    warnings = None
    info = None