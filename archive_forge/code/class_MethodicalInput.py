import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@attr.s(eq=False, hash=False)
class MethodicalInput(object):
    """
    An input for a L{MethodicalMachine}.
    """
    automaton = attr.ib(repr=False)
    method = attr.ib(validator=assertNoCode)
    symbol = attr.ib(repr=False)
    collectors = attr.ib(default=attr.Factory(dict), repr=False)
    argSpec = attr.ib(init=False, repr=False)

    @argSpec.default
    def _buildArgSpec(self):
        return _getArgSpec(self.method)

    def __get__(self, oself, type=None):
        """
        Return a function that takes no arguments and returns values returned
        by output functions produced by the given L{MethodicalInput} in
        C{oself}'s current state.
        """
        transitioner = _transitionerFromInstance(oself, self.symbol, self.automaton)

        @preserveName(self.method)
        @wraps(self.method)
        def doInput(*args, **kwargs):
            self.method(oself, *args, **kwargs)
            previousState = transitioner._state
            outputs, outTracer = transitioner.transition(self)
            collector = self.collectors[previousState]
            values = []
            for output in outputs:
                if outTracer:
                    outTracer(output._name())
                a, k = _filterArgs(args, kwargs, self.argSpec, output.argSpec)
                value = output(oself, *a, **k)
                values.append(value)
            return collector(values)
        return doInput

    def _name(self):
        return self.method.__name__