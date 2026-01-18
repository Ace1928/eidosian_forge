from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
@_SpecialForm
def Union(self, parameters):
    """Union type; Union[X, Y] means either X or Y.

    On Python 3.10 and higher, the | operator
    can also be used to denote unions;
    X | Y means the same thing to the type checker as Union[X, Y].

    To define a union, use e.g. Union[int, str]. Details:
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by
      type(None).
    - Unions of unions are flattened, e.g.::

        assert Union[Union[int, str], float] == Union[int, str, float]

    - Unions of a single argument vanish, e.g.::

        assert Union[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        assert Union[int, str, int] == Union[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        assert Union[int, str] == Union[str, int]

    - You cannot subclass or instantiate a union.
    - You can use Optional[X] as a shorthand for Union[X, None].
    """
    if parameters == ():
        raise TypeError('Cannot take a Union of no types.')
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    msg = 'Union[arg, ...]: each arg must be a type.'
    parameters = tuple((_type_check(p, msg) for p in parameters))
    parameters = _remove_dups_flatten(parameters)
    if len(parameters) == 1:
        return parameters[0]
    if len(parameters) == 2 and type(None) in parameters:
        return _UnionGenericAlias(self, parameters, name='Optional')
    return _UnionGenericAlias(self, parameters)