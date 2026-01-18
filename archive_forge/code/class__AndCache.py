from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class _AndCache(dict):
    """
    Cache for And operation between filters.
    (Filter classes are stateless, so we can reuse them.)

    Note: This could be a memory leak if we keep creating filters at runtime.
          If that is True, the filters should be weakreffed (not the tuple of
          filters), and tuples should be removed when one of these filters is
          removed. In practise however, there is a finite amount of filters.
    """

    def __missing__(self, filters):
        a, b = filters
        assert isinstance(b, Filter), 'Expecting filter, got %r' % b
        if isinstance(b, Always) or isinstance(a, Never):
            return a
        elif isinstance(b, Never) or isinstance(a, Always):
            return b
        result = _AndList(filters)
        self[filters] = result
        return result