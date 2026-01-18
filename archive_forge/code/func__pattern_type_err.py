from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def _pattern_type_err(self, pattern):
    raise TypeError('got {badtype} ({badobj!r}) as pattern, must be one of: {goodtypes}, pexpect.EOF, pexpect.TIMEOUT'.format(badtype=type(pattern), badobj=pattern, goodtypes=', '.join([str(ast) for ast in self.allowed_string_types])))