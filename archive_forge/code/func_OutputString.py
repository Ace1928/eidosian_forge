import re
import string
import types
def OutputString(self, attrs=None):
    result = []
    append = result.append
    append('%s=%s' % (self.key, self.coded_value))
    if attrs is None:
        attrs = self._reserved
    items = sorted(self.items())
    for key, value in items:
        if value == '':
            continue
        if key not in attrs:
            continue
        if key == 'expires' and isinstance(value, int):
            append('%s=%s' % (self._reserved[key], _getdate(value)))
        elif key == 'max-age' and isinstance(value, int):
            append('%s=%d' % (self._reserved[key], value))
        elif key == 'comment' and isinstance(value, str):
            append('%s=%s' % (self._reserved[key], _quote(value)))
        elif key in self._flags:
            if value:
                append(str(self._reserved[key]))
        else:
            append('%s=%s' % (self._reserved[key], value))
    return _semispacejoin(result)