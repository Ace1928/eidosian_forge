import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@argSpec.default
def _buildArgSpec(self):
    return _getArgSpec(self.method)