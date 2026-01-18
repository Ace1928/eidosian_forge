import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
def _copy_flag_dict(flag: flags.Flag) -> Mapping[str, Any]:
    """Returns a copy of the flag object's ``__dict__``.

  It's mostly a shallow copy of the ``__dict__``, except it also does a shallow
  copy of the validator list.

  Args:
    flag: flags.Flag, the flag to copy.

  Returns:
    A copy of the flag object's ``__dict__``.
  """
    copy = flag.__dict__.copy()
    copy['_value'] = flag.value
    copy['validators'] = list(flag.validators)
    return copy