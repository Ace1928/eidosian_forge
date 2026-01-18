from collections import defaultdict, namedtuple, OrderedDict
from functools import wraps
from itertools import product
import os
import types
import warnings
from .. import __url__
from .deprecation import Deprecation
class defaultkeydict(defaultdict):
    """defaultdict where default_factory should have the signature key -> value

    Examples
    --------
    >>> d = defaultkeydict(lambda k: '[%s]' % k, {'a': '[a]', 'b': '[B]'})
    >>> d['a']
    '[a]'
    >>> d['b']
    '[B]'
    >>> d['c']
    '[c]'

    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError('Missing key: %s' % key)
        else:
            self[key] = self.default_factory(key)
        return self[key]