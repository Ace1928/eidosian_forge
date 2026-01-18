import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
def _oneTransition(self, startState, inputToken, endState, outputTokens, collector):
    """
        See L{MethodicalState.upon}.
        """
    self._automaton.addTransition(startState, inputToken, endState, tuple(outputTokens))
    inputToken.collectors[startState] = collector