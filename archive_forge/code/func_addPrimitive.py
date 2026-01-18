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
def addPrimitive(self, primitive, arity, name=None):
    """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
    assert arity > 0, 'arity should be >= 1'
    args = [__type__] * arity
    PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)