from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
class juxt(object):
    """ Creates a function that calls several functions with the same arguments

    Takes several functions and returns a function that applies its arguments
    to each of those functions then returns a tuple of the results.

    Name comes from juxtaposition: the fact of two things being seen or placed
    close together with contrasting effect.

    >>> inc = lambda x: x + 1
    >>> double = lambda x: x * 2
    >>> juxt(inc, double)(10)
    (11, 20)
    >>> juxt([inc, double])(10)
    (11, 20)
    """
    __slots__ = ['funcs']

    def __init__(self, *funcs):
        if len(funcs) == 1 and (not callable(funcs[0])):
            funcs = funcs[0]
        self.funcs = tuple(funcs)

    def __call__(self, *args, **kwargs):
        return tuple((func(*args, **kwargs) for func in self.funcs))

    def __getstate__(self):
        return self.funcs

    def __setstate__(self, state):
        self.funcs = state