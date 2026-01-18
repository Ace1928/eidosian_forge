import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@attr.s(frozen=True)
class MethodicalOutput(object):
    """
    An output for a L{MethodicalMachine}.
    """
    machine = attr.ib(repr=False)
    method = attr.ib()
    argSpec = attr.ib(init=False, repr=False)

    @argSpec.default
    def _buildArgSpec(self):
        return _getArgSpec(self.method)

    def __get__(self, oself, type=None):
        """
        Outputs are private, so raise an exception when we attempt to get one.
        """
        raise AttributeError('{cls}.{method} is a state-machine output method; to produce this output, call an input method instead.'.format(cls=type.__name__, method=self.method.__name__))

    def __call__(self, oself, *args, **kwargs):
        """
        Call the underlying method.
        """
        return self.method(oself, *args, **kwargs)

    def _name(self):
        return self.method.__name__