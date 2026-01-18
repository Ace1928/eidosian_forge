from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class _OrList(Filter):
    """
    Result of |-operation between several filters.
    """

    def __init__(self, filters):
        all_filters = []
        for f in filters:
            if isinstance(f, _OrList):
                all_filters.extend(f.filters)
            else:
                all_filters.append(f)
        self.filters = all_filters

    def test_args(self, *args):
        return all((f.test_args(*args) for f in self.filters))

    def __call__(self, *a, **kw):
        return any((f(*a, **kw) for f in self.filters))

    def __repr__(self):
        return '|'.join((repr(f) for f in self.filters))