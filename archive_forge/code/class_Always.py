from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class Always(Filter):
    """
    Always enable feature.
    """

    def __call__(self, *a, **kw):
        return True

    def __invert__(self):
        return Never()