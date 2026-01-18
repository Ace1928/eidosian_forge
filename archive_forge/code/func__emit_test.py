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
def _emit_test():
    """write out a test"""
    if test_name is None:
        return
    if log:
        log_bytes = b'\n'.join((log_line.encode('utf8') for log_line in log))
        mime_type = UTF8_TEXT
        file_name = 'tap comment'
        eof = True
    else:
        log_bytes = None
        mime_type = None
        file_name = None
        eof = True
    del log[:]
    output.status(test_id=test_name, test_status=result, file_bytes=log_bytes, mime_type=mime_type, eof=eof, file_name=file_name, runnable=False)