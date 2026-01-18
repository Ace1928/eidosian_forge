import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def _construct_overrider(flag_overrider_cls, *args, **kwargs):
    """Handles the args/kwargs returning an instance of flag_overrider_cls.

  If flag_overrider_cls is _FlagOverrider then values should be native python
  types matching the python types. Otherwise if flag_overrider_cls is
  _ParsingFlagOverrider the values should be strings or sequences of strings.

  Args:
    flag_overrider_cls: The class that will do the overriding.
    *args: Tuples of FlagHolder and the new flag value.
    **kwargs: Keword args mapping flag name to new flag value.

  Returns:
    A _FlagOverrider to be used as a decorator or context manager.
  """
    if not args:
        return flag_overrider_cls(**kwargs)
    if len(args) == 1 and callable(args[0]):
        if kwargs:
            raise ValueError("It's invalid to specify both positional and keyword parameters.")
        func = args[0]
        if inspect.isclass(func):
            raise TypeError('@flagsaver.flagsaver cannot be applied to a class.')
        return _wrap(flag_overrider_cls, func, {})
    for arg in args:
        if not isinstance(arg, tuple) or len(arg) != 2:
            raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
        holder, value = arg
        if not isinstance(holder, flags.FlagHolder):
            raise ValueError('Expected (FlagHolder, value) pair, found %r' % (arg,))
        if holder.name in kwargs:
            raise ValueError('Cannot set --%s multiple times' % holder.name)
        kwargs[holder.name] = value
    return flag_overrider_cls(**kwargs)