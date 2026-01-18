import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@attr.s(eq=False, hash=False)
class MethodicalTracer(object):
    automaton = attr.ib(repr=False)
    symbol = attr.ib(repr=False)

    def __get__(self, oself, type=None):
        transitioner = _transitionerFromInstance(oself, self.symbol, self.automaton)

        def setTrace(tracer):
            transitioner.setTrace(tracer)
        return setTrace