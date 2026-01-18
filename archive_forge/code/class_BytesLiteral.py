from __future__ import absolute_import
import re
import sys
class BytesLiteral(_bytes):
    encoding = None

    def __deepcopy__(self, memo):
        return self

    def byteencode(self):
        if IS_PYTHON3:
            return _bytes(self)
        else:
            return self.decode('ISO-8859-1').encode('ISO-8859-1')

    def utf8encode(self):
        assert False, 'this is not a unicode string: %r' % self

    def __str__(self):
        """Fake-decode the byte string to unicode to support %
        formatting of unicode strings.
        """
        return self.decode('ISO-8859-1')
    is_unicode = False

    def as_c_string_literal(self):
        value = split_string_literal(escape_byte_string(self))
        return '"%s"' % value
    if not hasattr(_bytes, 'isascii'):

        def isascii(self):
            return True