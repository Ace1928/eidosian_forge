import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@_ExtensionsSpecialForm
def TypeAlias(self, parameters):
    """Special marker indicating that an assignment should
        be recognized as a proper type alias definition by type
        checkers.

        For example::

            Predicate: TypeAlias = Callable[..., bool]

        It's invalid when used anywhere except as in the example above.
        """
    raise TypeError(f'{self} is not subscriptable')