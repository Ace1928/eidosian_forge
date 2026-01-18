from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Any
import warnings
from typing import TYPE_CHECKING
def _check_param_one_of(self, param: str, options: Iterable[Any]) -> None:
    """Raise when parameter value is not one of a specified set."""
    value = getattr(self, param)
    if value not in options:
        *most, last = options
        option_str = ', '.join((f'{x!r}' for x in most[:-1])) + f' or {last!r}'
        err = ' '.join([f'The `{param}` parameter for `{self.__class__.__name__}` must be', f'one of {option_str}; not {value!r}.'])
        raise ValueError(err)