from pathlib import Path
from numpy.lib._iotools import _is_string_like
class EmptyContextManager:
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """

    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        """When entering, return the embedded object"""
        return self._obj

    def __exit__(self, *args):
        """Do not hide anything"""
        return False

    def __getattr__(self, name):
        return getattr(self._obj, name)