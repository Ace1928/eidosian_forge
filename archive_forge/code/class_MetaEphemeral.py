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
class MetaEphemeral(type):
    """Meta-Class that creates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated.
    """
    cache = {}

    def __new__(meta, name, func, ret=__type__, id_=None):
        if id_ in MetaEphemeral.cache:
            return MetaEphemeral.cache[id_]
        if isinstance(func, types.LambdaType) and func.__name__ == '<lambda>':
            warnings.warn('Ephemeral {name} function cannot be pickled because its generating function is a lambda function. Use functools.partial instead.'.format(name=name), RuntimeWarning)

        def __init__(self):
            self.value = func()
        attr = {'__init__': __init__, 'name': name, 'func': func, 'ret': ret, 'conv_fct': repr}
        cls = super(MetaEphemeral, meta).__new__(meta, name, (Terminal,), attr)
        MetaEphemeral.cache[id(cls)] = cls
        return cls

    def __init__(cls, name, func, ret=__type__, id_=None):
        super(MetaEphemeral, cls).__init__(name, (Terminal,), {})

    def __reduce__(cls):
        return (MetaEphemeral, (cls.name, cls.func, cls.ret, id(cls)))