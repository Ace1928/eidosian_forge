from __future__ import print_function, unicode_literals
import itertools
from collections import OrderedDict, deque
from functools import wraps
from types import GeneratorType
from six.moves import zip_longest
from .py3compat import fix_unicode_literals_in_doctest
class CaseInsensitiveDefaultDict(CaseInsensitiveDict):
    """CaseInseisitiveDict with default factory, like collections.defaultdict

    >>> d = CaseInsensitiveDefaultDict(int)
    >>> d['a']
    0
    >>> d['a'] += 1
    >>> d['a']
    1
    >>> d['A']
    1
    >>> d['a'] = 3
    >>> d['a']
    3
    >>> d['B'] += 10
    >>> d['b']
    10

    """

    def __init__(self, default_factory):
        super(CaseInsensitiveDefaultDict, self).__init__()
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super(CaseInsensitiveDefaultDict, self).__getitem__(key)
        except KeyError:
            return self.default_factory()