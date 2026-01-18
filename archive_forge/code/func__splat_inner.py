import collections.abc
import functools
import inspect
import itertools
import operator
import time
import types
import warnings
import more_itertools
@functools.singledispatch
def _splat_inner(args, func):
    """Splat args to func."""
    return func(*args)