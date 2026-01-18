import ast as _ast
import io as _io
import os as _os
import collections.abc
def _verify_open(self):
    if self._index is None:
        raise error('DBM object has already been closed')