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
class PyUnicodeObjectPtr(PyObjectPtr):
    _typename = 'PyUnicodeObject'

    def char_width(self):
        _type_Py_UNICODE = gdb.lookup_type('Py_UNICODE')
        return _type_Py_UNICODE.sizeof

    def proxyval(self, visited):
        global _is_pep393
        if _is_pep393 is None:
            fields = gdb.lookup_type('PyUnicodeObject').fields()
            _is_pep393 = 'data' in [f.name for f in fields]
        if _is_pep393:
            may_have_surrogates = False
            compact = self.field('_base')
            ascii = compact['_base']
            state = ascii['state']
            is_compact_ascii = int(state['ascii']) and int(state['compact'])
            if not int(state['ready']):
                field_length = long(compact['wstr_length'])
                may_have_surrogates = True
                field_str = ascii['wstr']
            else:
                field_length = long(ascii['length'])
                if is_compact_ascii:
                    field_str = ascii.address + 1
                elif int(state['compact']):
                    field_str = compact.address + 1
                else:
                    field_str = self.field('data')['any']
                repr_kind = int(state['kind'])
                if repr_kind == 1:
                    field_str = field_str.cast(_type_unsigned_char_ptr())
                elif repr_kind == 2:
                    field_str = field_str.cast(_type_unsigned_short_ptr())
                elif repr_kind == 4:
                    field_str = field_str.cast(_type_unsigned_int_ptr())
        else:
            field_length = long(self.field('length'))
            field_str = self.field('str')
            may_have_surrogates = self.char_width() == 2
        if not may_have_surrogates:
            Py_UNICODEs = [int(field_str[i]) for i in safe_range(field_length)]
        else:
            Py_UNICODEs = []
            i = 0
            limit = safety_limit(field_length)
            while i < limit:
                ucs = int(field_str[i])
                i += 1
                if ucs < 55296 or ucs >= 56320 or i == field_length:
                    Py_UNICODEs.append(ucs)
                    continue
                ucs2 = int(field_str[i])
                if ucs2 < 56320 or ucs2 > 57343:
                    continue
                code = (ucs & 1023) << 10
                code |= ucs2 & 1023
                code += 65536
                Py_UNICODEs.append(code)
                i += 1
        result = u''.join([_unichr(ucs) if ucs <= 1114111 else '�' for ucs in Py_UNICODEs])
        return result

    def write_repr(self, out, visited):
        proxy = self.proxyval(visited)
        if "'" in proxy and '"' not in proxy:
            quote = '"'
        else:
            quote = "'"
        out.write(quote)
        i = 0
        while i < len(proxy):
            ch = proxy[i]
            i += 1
            if ch == quote or ch == '\\':
                out.write('\\')
                out.write(ch)
            elif ch == '\t':
                out.write('\\t')
            elif ch == '\n':
                out.write('\\n')
            elif ch == '\r':
                out.write('\\r')
            elif ch < ' ' or ch == 127:
                out.write('\\x')
                out.write(hexdigits[ord(ch) >> 4 & 15])
                out.write(hexdigits[ord(ch) & 15])
            elif ord(ch) < 127:
                out.write(ch)
            else:
                ucs = ch
                ch2 = None
                if sys.maxunicode < 65536:
                    if i < len(proxy) and 55296 <= ord(ch) < 56320 and (56320 <= ord(proxy[i]) <= 57343):
                        ch2 = proxy[i]
                        ucs = ch + ch2
                        i += 1
                printable = _unichr_is_printable(ucs)
                if printable:
                    try:
                        ucs.encode(ENCODING)
                    except UnicodeEncodeError:
                        printable = False
                if not printable:
                    if ch2 is not None:
                        code = (ord(ch) & 1023) << 10
                        code |= ord(ch2) & 1023
                        code += 65536
                    else:
                        code = ord(ucs)
                    if code <= 255:
                        out.write('\\x')
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                    elif code >= 65536:
                        out.write('\\U')
                        out.write(hexdigits[code >> 28 & 15])
                        out.write(hexdigits[code >> 24 & 15])
                        out.write(hexdigits[code >> 20 & 15])
                        out.write(hexdigits[code >> 16 & 15])
                        out.write(hexdigits[code >> 12 & 15])
                        out.write(hexdigits[code >> 8 & 15])
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                    else:
                        out.write('\\u')
                        out.write(hexdigits[code >> 12 & 15])
                        out.write(hexdigits[code >> 8 & 15])
                        out.write(hexdigits[code >> 4 & 15])
                        out.write(hexdigits[code & 15])
                else:
                    out.write(ch)
                    if ch2 is not None:
                        out.write(ch2)
        out.write(quote)