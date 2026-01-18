from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
class SafeList(list):
    """List with a clear() method.

    .. warning::
        Usage of the iterate() function will decrease your performance.
    """

    @deprecated
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clear(self):
        del self[:]

    @deprecated
    def iterate(self, reverse=False):
        if reverse:
            return iter(reversed(self))
        return iter(self)