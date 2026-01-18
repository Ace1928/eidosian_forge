from inspect import Parameter, signature
import functools
import warnings
from importlib import import_module
class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """

    def __init__(self, content, message):
        self._content = content
        self._message = message

    def __hash__(self):
        return hash(self._content)

    def __eq__(self, other):
        res = self._content == other
        if res:
            warnings.warn(self._message, category=DeprecationWarning, stacklevel=2)
        return res