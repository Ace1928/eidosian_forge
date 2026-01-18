import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _print_string_literal_in_array(self, s):
    prnt = self._prnt
    prnt('// # NB. this is not a string because of a size limit in MSVC')
    if not isinstance(s, bytes):
        s = s.encode('utf-8')
    else:
        s.decode('utf-8')
    try:
        s.decode('ascii')
    except UnicodeDecodeError:
        s = b'# -*- encoding: utf8 -*-\n' + s
    for line in s.splitlines(True):
        comment = line
        if type('//') is bytes:
            line = map(ord, line)
        else:
            comment = ascii(comment)[1:-1]
        prnt(('// ' + comment).rstrip())
        printed_line = ''
        for c in line:
            if len(printed_line) >= 76:
                prnt(printed_line)
                printed_line = ''
            printed_line += '%d,' % (c,)
        prnt(printed_line)