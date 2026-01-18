from collections import defaultdict, namedtuple, OrderedDict
from functools import wraps
from itertools import product
import os
import types
import warnings
from .. import __url__
from .deprecation import Deprecation
class NameSpace:
    """Used to wrap, e.g. modules.

    Parameters
    ----------
    default : module
        The underlying module. Acts as a fallback for attribute access.

    Examples
    --------
    >>> import numpy
    >>> my_numpy = NameSpace(numpy)
    >>> my_numpy.array = lambda *args, **kwargs: list(numpy.array(*args, **kwargs))
    >>> isinstance(my_numpy.array([2, 3]), list)
    True
    >>> isinstance(numpy.array([2, 3]), list)
    False

    """

    def __init__(self, default):
        self._NameSpace_default = default
        self._NameSpace_attr_store = {}

    def __getattr__(self, attr):
        if attr.startswith('_NameSpace_'):
            return self.__dict__[attr]
        else:
            try:
                return self._NameSpace_attr_store[attr]
            except KeyError:
                return getattr(self._NameSpace_default, attr)

    def __setattr__(self, attr, val):
        if attr.startswith('_NameSpace_'):
            self.__dict__[attr] = val
        else:
            self._NameSpace_attr_store[attr] = val

    def as_dict(self):
        items = self._NameSpace_default.__dict__.items()
        result = {k: v for k, v in items if not k.startswith('_')}
        result.update(self._NameSpace_attr_store)
        return result