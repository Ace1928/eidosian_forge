import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def _get_text_attributes():
    return win32.GetConsoleScreenBufferInfo().wAttributes