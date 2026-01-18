import re
import string
import types
class SimpleCookie(BaseCookie):
    """
    SimpleCookie supports strings as cookie values.  When setting
    the value using the dictionary assignment notation, SimpleCookie
    calls the builtin str() to convert the value to a string.  Values
    received from HTTP are kept as strings.
    """

    def value_decode(self, val):
        return (_unquote(val), val)

    def value_encode(self, val):
        strval = str(val)
        return (strval, _quote(strval))