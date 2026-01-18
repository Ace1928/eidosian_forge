import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
class AddX(object):

    def __init__(self, func):
        self.func = func

    def __call__(self, addx, *args, **kwargs):
        return addx + self.func(*args, **kwargs)

    @property
    def __signature__(self):
        sig = inspect.signature(self.func)
        params = list(sig.parameters.values())
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        newparam = inspect.Parameter('addx', kind)
        params = [newparam] + params
        return sig.replace(parameters=params)