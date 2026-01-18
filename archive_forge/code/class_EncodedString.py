from __future__ import absolute_import
import re
import sys
class EncodedString(_unicode):
    encoding = None

    def __deepcopy__(self, memo):
        return self

    def byteencode(self):
        assert self.encoding is not None
        return self.encode(self.encoding)

    def utf8encode(self):
        assert self.encoding is None
        return self.encode('UTF-8')

    @property
    def is_unicode(self):
        return self.encoding is None

    def contains_surrogates(self):
        return string_contains_surrogates(self)

    def as_utf8_string(self):
        return bytes_literal(self.utf8encode(), 'utf8')

    def as_c_string_literal(self):
        if self.encoding is None:
            s = self.as_utf8_string()
        else:
            s = bytes_literal(self.byteencode(), self.encoding)
        return s.as_c_string_literal()
    if not hasattr(_unicode, 'isascii'):

        def isascii(self):
            try:
                self.encode('ascii')
            except UnicodeEncodeError:
                return False
            else:
                return True