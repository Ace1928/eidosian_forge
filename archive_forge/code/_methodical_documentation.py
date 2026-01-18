import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName

        Generate a L{graphviz.Digraph} that represents this machine's
        states and transitions.

        @return: L{graphviz.Digraph} object; for more information, please
            see the documentation for
            U{graphviz<https://graphviz.readthedocs.io/>}

        