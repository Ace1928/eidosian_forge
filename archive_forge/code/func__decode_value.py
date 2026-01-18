import cgi
import copy
import sys
from collections.abc import MutableMapping as DictMixin
def _decode_value(self, value):
    """
        Decode the specified value to unicode. Assumes value is a ``str`` or
        `FieldStorage`` object.

        ``FieldStorage`` objects are specially handled.
        """
    if isinstance(value, cgi.FieldStorage):
        decode_name = self.decode_keys and isinstance(value.name, bytes)
        if decode_name:
            value = copy.copy(value)
            if decode_name:
                value.name = value.name.decode(self.encoding, self.errors)
    else:
        try:
            value = value.decode(self.encoding, self.errors)
        except AttributeError:
            pass
    return value