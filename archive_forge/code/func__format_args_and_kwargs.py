import functools
import inspect
import reprlib
import sys
import traceback
from . import constants
def _format_args_and_kwargs(args, kwargs):
    """Format function arguments and keyword arguments.

    Special case for a single parameter: ('hello',) is formatted as ('hello').
    """
    items = []
    if args:
        items.extend((reprlib.repr(arg) for arg in args))
    if kwargs:
        items.extend((f'{k}={reprlib.repr(v)}' for k, v in kwargs.items()))
    return '({})'.format(', '.join(items))