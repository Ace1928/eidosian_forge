import ast as _ast
import io as _io
import os as _os
import collections.abc
def _setval(self, pos, val):
    with _io.open(self._datfile, 'rb+') as f:
        f.seek(pos)
        f.write(val)
    return (pos, len(val))