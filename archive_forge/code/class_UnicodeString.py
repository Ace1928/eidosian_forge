import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class UnicodeString(ByteString):
    """Convert things to unicode string.

    This is implemented as a specialization of the ByteString class.

    You can also use the alias `String` for this validator.

    In addition to the String arguments, an encoding argument is also
    accepted. By default, the encoding will be utf-8. You can overwrite
    this using the encoding parameter. You can also set inputEncoding
    and outputEncoding differently. An inputEncoding of None means
    "do not decode", an outputEncoding of None means "do not encode".

    All converted strings are returned as Unicode strings.

    ::

        >>> UnicodeString().to_python(None) == ''
        True
        >>> UnicodeString().to_python([]) == ''
        True
        >>> UnicodeString(encoding='utf-7').to_python('Ni Ni Ni') == 'Ni Ni Ni'
        True

    """
    encoding = 'utf-8'
    inputEncoding = NoDefault
    outputEncoding = NoDefault
    messages = dict(badEncoding=_('Invalid data or incorrect encoding'))

    def __init__(self, **kw):
        ByteString.__init__(self, **kw)
        if self.inputEncoding is NoDefault:
            self.inputEncoding = self.encoding
        if self.outputEncoding is NoDefault:
            self.outputEncoding = self.encoding

    def _convert_to_python(self, value, state):
        if not value:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode(encoding=self.inputEncoding or self.encoding)
            except UnicodeDecodeError:
                raise Invalid(self.message('badEncoding', state), value, state)
        return str(value)

    def _convert_from_python(self, value, state):
        if not isinstance(value, str):
            value = str(value)
        if self.outputEncoding and isinstance(value, str):
            value = value.encode(self.outputEncoding)
        return value

    def empty_value(self, value):
        return ''