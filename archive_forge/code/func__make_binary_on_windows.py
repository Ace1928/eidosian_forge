import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def _make_binary_on_windows(fileno):
    """Win32 mangles \r
 to 
 and that breaks streams. See bug lp:505078."""
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.setmode(fileno, os.O_BINARY)