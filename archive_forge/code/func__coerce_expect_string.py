from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def _coerce_expect_string(self, s):
    if self.encoding is None and (not isinstance(s, bytes)):
        return s.encode('ascii')
    return s