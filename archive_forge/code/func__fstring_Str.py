from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO
def _fstring_Str(self, t, write):
    value = t.s.replace('{', '{{').replace('}', '}}')
    write(value)