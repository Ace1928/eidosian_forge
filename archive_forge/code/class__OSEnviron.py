import os
from .dependencies import ctypes
class _OSEnviron(object):
    """Helper class to proxy a "DLL-like" interface to os.environ"""
    _libname = 'os.environ'

    def available(self):
        return True

    def get_env_dict(self):
        return dict(os.environ)

    def getenv(self, key):
        try:
            return os.environb.get(key, None)
        except AttributeError:
            return _as_bytes(os.environ.get(_as_unicode(key), None))

    def wgetenv(self, key):
        return _as_unicode(os.environ.get(key, None))

    def putenv_s(self, key, val):
        if not val:
            if key in os.environ:
                del os.environ[key]
            return
        os.environb[key] = val

    def wputenv_s(self, key, val):
        if not val:
            if key in os.environ:
                del os.environ[key]
            return
        os.environ[key] = val