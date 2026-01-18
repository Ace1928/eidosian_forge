import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class _DecoratorBaseClass:
    """Used to manage decorators' warnings stacklevel.

    The `_stack_length` class variable is used to store the number of
    times a function is wrapped by a decorator.

    Let `stack_length` be the total number of times a decorated
    function is wrapped, and `stack_rank` be the rank of the decorator
    in the decorators stack. The stacklevel of a warning is then
    `stacklevel = 1 + stack_length - stack_rank`.
    """
    _stack_length = {}

    def get_stack_length(self, func):
        length = self._stack_length.get(func.__name__, _get_stack_length(func))
        return length