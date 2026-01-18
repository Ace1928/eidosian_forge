import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix='ARG'):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, 'arity should be >= 1'
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)

    def addTerminal(self, terminal, name=None):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)