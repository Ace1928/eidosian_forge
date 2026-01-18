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
def _unwrap_text(stream):
    """Unwrap stream if it is a text stream to get the original buffer."""
    exceptions = (_UnsupportedOperation, IOError)
    unicode_type = str
    try:
        if type(stream.read(0)) is unicode_type:
            return stream.buffer
    except exceptions:
        try:
            stream.write(_b(''))
        except TypeError:
            return stream.buffer
    return stream