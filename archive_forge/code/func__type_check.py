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
def _type_check(arg, msg, is_argument=True, module=None, *, allow_special_forms=False):
    """Check that the argument is a type, and return it (internal helper).

    As a special case, accept None and return type(None) instead. Also wrap strings
    into ForwardRef instances. Consider several corner cases, for example plain
    special forms like Union are not valid, while Union[int, str] is OK, etc.
    The msg argument is a human-readable error message, e.g.::

        "Union[arg, ...]: arg should be a type."

    We append the repr() of the actual value (truncated to 100 chars).
    """
    invalid_generic_forms = (Generic, Protocol)
    if not allow_special_forms:
        invalid_generic_forms += (ClassVar,)
        if is_argument:
            invalid_generic_forms += (Final,)
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    if isinstance(arg, _GenericAlias) and arg.__origin__ in invalid_generic_forms:
        raise TypeError(f'{arg} is not valid as type argument')
    if arg in (Any, LiteralString, NoReturn, Never, Self, TypeAlias):
        return arg
    if allow_special_forms and arg in (ClassVar, Final):
        return arg
    if isinstance(arg, _SpecialForm) or arg in (Generic, Protocol):
        raise TypeError(f'Plain {arg} is not valid as type argument')
    if type(arg) is tuple:
        raise TypeError(f'{msg} Got {arg!r:.100}.')
    return arg