from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PyBytesObjectPtr(PyObjectPtr):
    _typename = 'PyBytesObject'

    def __str__(self):
        field_ob_size = self.field('ob_size')
        field_ob_sval = self.field('ob_sval')
        char_ptr = field_ob_sval.address.cast(_type_unsigned_char_ptr())
        return ''.join([chr(char_ptr[i]) for i in safe_range(field_ob_size)])

    def proxyval(self, visited):
        return str(self)

    def write_repr(self, out, visited):
        proxy = self.proxyval(visited)
        quote = "'"
        if "'" in proxy and (not '"' in proxy):
            quote = '"'
        out.write('b')
        out.write(quote)
        for byte in proxy:
            if byte == quote or byte == '\\':
                out.write('\\')
                out.write(byte)
            elif byte == '\t':
                out.write('\\t')
            elif byte == '\n':
                out.write('\\n')
            elif byte == '\r':
                out.write('\\r')
            elif byte < ' ' or ord(byte) >= 127:
                out.write('\\x')
                out.write(hexdigits[(ord(byte) & 240) >> 4])
                out.write(hexdigits[ord(byte) & 15])
            else:
                out.write(byte)
        out.write(quote)