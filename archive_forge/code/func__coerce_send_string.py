from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def _coerce_send_string(self, s):
    if self.encoding is None and (not isinstance(s, bytes)):
        return s.encode('utf-8')
    return s