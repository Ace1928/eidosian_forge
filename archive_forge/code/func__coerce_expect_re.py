from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def _coerce_expect_re(self, r):
    p = r.pattern
    if self.encoding is None and (not isinstance(p, bytes)):
        return re.compile(p.encode('utf-8'))
    elif self.encoding is not None and isinstance(p, bytes):
        return re.compile(p.decode('utf-8'))
    return r