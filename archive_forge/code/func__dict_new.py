import abc
import collections
import collections.abc
import operator
import sys
import typing
def _dict_new(*args, **kwargs):
    if not args:
        raise TypeError('TypedDict.__new__(): not enough arguments')
    _, args = (args[0], args[1:])
    return dict(*args, **kwargs)