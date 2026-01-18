from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class _Invert(Filter):
    """
    Negation of another filter.
    """

    def __init__(self, filter):
        self.filter = filter

    def __call__(self, *a, **kw):
        return not self.filter(*a, **kw)

    def __repr__(self):
        return '~%r' % self.filter

    def test_args(self, *args):
        return self.filter.test_args(*args)