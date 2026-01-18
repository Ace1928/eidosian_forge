from __future__ import (absolute_import, division, print_function)
import os
import sys
from collections.abc import MutableMapping
from ansible.module_utils.six import PY3
from ansible.module_utils.common.text.converters import to_bytes, to_text
class _TextEnviron(MutableMapping):
    """
    Utility class to return text strings from the environment instead of byte strings

    Mimics the behaviour of os.environ on Python3
    """

    def __init__(self, env=None, encoding=None):
        if env is None:
            env = os.environ
        self._raw_environ = env
        self._value_cache = {}
        if encoding is None:
            self.encoding = sys.getfilesystemencoding()
        else:
            self.encoding = encoding

    def __delitem__(self, key):
        del self._raw_environ[key]

    def __getitem__(self, key):
        value = self._raw_environ[key]
        if PY3:
            return value
        if value not in self._value_cache:
            self._value_cache[value] = to_text(value, encoding=self.encoding, nonstring='passthru', errors='surrogate_or_strict')
        return self._value_cache[value]

    def __setitem__(self, key, value):
        self._raw_environ[key] = to_bytes(value, encoding=self.encoding, nonstring='strict', errors='surrogate_or_strict')

    def __iter__(self):
        return self._raw_environ.__iter__()

    def __len__(self):
        return len(self._raw_environ)