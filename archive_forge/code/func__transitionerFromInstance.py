import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
def _transitionerFromInstance(oself, symbol, automaton):
    """
    Get a L{Transitioner}
    """
    transitioner = getattr(oself, symbol, None)
    if transitioner is None:
        transitioner = Transitioner(automaton, automaton.initialState)
        setattr(oself, symbol, transitioner)
    return transitioner