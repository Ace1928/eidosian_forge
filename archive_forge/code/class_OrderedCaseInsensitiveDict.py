from __future__ import print_function, unicode_literals
import itertools
from collections import OrderedDict, deque
from functools import wraps
from types import GeneratorType
from six.moves import zip_longest
from .py3compat import fix_unicode_literals_in_doctest
@fix_unicode_literals_in_doctest
class OrderedCaseInsensitiveDict(CaseInsensitiveDict):
    """ An (incomplete) ordered case-insensitive dict.

    >>> d = OrderedCaseInsensitiveDict([
    ...     ('Uno', 1),
    ...     ('Dos', 2),
    ...     ('Tres', 3),
    ... ])
    >>> d
    OrderedCaseInsensitiveDict([(u'Uno', 1), (u'Dos', 2), (u'Tres', 3)])
    >>> d.lower()
    OrderedCaseInsensitiveDict([(u'uno', 1), (u'dos', 2), (u'tres', 3)])
    >>> list(d.keys())
    [u'Uno', u'Dos', u'Tres']
    >>> list(d.items())
    [(u'Uno', 1), (u'Dos', 2), (u'Tres', 3)]
    >>> list(d.values())
    [1, 2, 3]
    >>> d['Cuatro'] = 4
    >>> list(d.keys())
    [u'Uno', u'Dos', u'Tres', u'Cuatro']
    >>> list(d.items())
    [(u'Uno', 1), (u'Dos', 2), (u'Tres', 3), (u'Cuatro', 4)]
    >>> list(d.values())
    [1, 2, 3, 4]
    >>> list(d)
    [u'Uno', u'Dos', u'Tres', u'Cuatro']
    >>> 'Uno' in d
    True
    >>> 'uno' in d
    True
    >>> d['Uno']
    1
    >>> d['uno']
    1
    >>> d['UNO']
    1
    >>> 'Cuatro' in d
    True
    >>> 'CUATRO' in d
    True
    >>> d['Cuatro']
    4
    >>> d['cuatro']
    4
    >>> d['UNO'] = 'one'
    >>> d['uno']
    u'one'
    >>> d['Uno']
    u'one'
    >>> list(d.keys())
    [u'UNO', u'Dos', u'Tres', u'Cuatro']
    >>> d['cuatro'] = 'four'
    >>> d['Cuatro']
    u'four'
    >>> d['cuatro']
    u'four'
    >>> list(d.keys())
    [u'UNO', u'Dos', u'Tres', u'cuatro']
    >>> list(d.values())
    [u'one', 2, 3, u'four']
    >>> del d['dos']
    >>> list(d.keys())
    [u'UNO', u'Tres', u'cuatro']
    >>> list(d.values())
    [u'one', 3, u'four']
    """

    def __init__(self, *args, **kwargs):
        initial = OrderedDict(*args, **kwargs)
        self._dict = dict(((key.lower(), value) for key, value in initial.items()))
        self._keys = OrderedDict(((key.lower(), key) for key in initial))

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, list(self.items()))