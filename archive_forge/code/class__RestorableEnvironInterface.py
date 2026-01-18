import os
from .dependencies import ctypes
class _RestorableEnvironInterface(object):
    """Interface to track environment changes and restore state"""

    def __init__(self, dll):
        assert dll.available()
        self.dll = dll
        self._original_state = {}
        for key, val in list(os.environ.items()):
            if val != self[key]:
                self[key] = val
        origEnv = self.dll.get_env_dict()
        if origEnv is not None:
            for key in origEnv:
                if key not in os.environ:
                    del self[key]

    def restore(self):
        for key, val in self._original_state.items():
            if not val:
                if self[key] is not None:
                    del self[key]
            else:
                self[key] = val
        self._original_state = {}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dll.wgetenv(key)
        else:
            return self.dll.getenv(key)

    def __setitem__(self, key, val):
        if key not in self._original_state:
            self._original_state[key] = self[key]
        if isinstance(key, str):
            if isinstance(val, str):
                self.dll.wputenv_s(key, val)
            else:
                self.dll.wputenv_s(key, _as_unicode(val))
        elif isinstance(val, str):
            self.dll.wputenv_s(_as_unicode(key), val)
        else:
            self.dll.putenv_s(key, val)

    def __delitem__(self, key):
        if key not in self._original_state:
            self._original_state[key] = self[key]
        if isinstance(key, str):
            self.dll.wputenv_s(key, u'')
        else:
            self.dll.putenv_s(key, b'')