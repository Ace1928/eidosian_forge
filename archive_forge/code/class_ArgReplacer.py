import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
class ArgReplacer(object):
    """Replaces one value in an ``args, kwargs`` pair.

    Inspects the function signature to find an argument by name
    whether it is passed by position or keyword.  For use in decorators
    and similar wrappers.
    """

    def __init__(self, func: Callable, name: str) -> None:
        self.name = name
        try:
            self.arg_pos = self._getargnames(func).index(name)
        except ValueError:
            self.arg_pos = None

    def _getargnames(self, func: Callable) -> List[str]:
        try:
            return getfullargspec(func).args
        except TypeError:
            if hasattr(func, 'func_code'):
                code = func.func_code
                return code.co_varnames[:code.co_argcount]
            raise

    def get_old_value(self, args: Sequence[Any], kwargs: Dict[str, Any], default: Any=None) -> Any:
        """Returns the old value of the named argument without replacing it.

        Returns ``default`` if the argument is not present.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            return args[self.arg_pos]
        else:
            return kwargs.get(self.name, default)

    def replace(self, new_value: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Tuple[Any, Sequence[Any], Dict[str, Any]]:
        """Replace the named argument in ``args, kwargs`` with ``new_value``.

        Returns ``(old_value, args, kwargs)``.  The returned ``args`` and
        ``kwargs`` objects may not be the same as the input objects, or
        the input objects may be mutated.

        If the named argument was not found, ``new_value`` will be added
        to ``kwargs`` and None will be returned as ``old_value``.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            old_value = args[self.arg_pos]
            args = list(args)
            args[self.arg_pos] = new_value
        else:
            old_value = kwargs.get(self.name)
            kwargs[self.name] = new_value
        return (old_value, args, kwargs)