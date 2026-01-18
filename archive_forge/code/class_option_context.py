from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> from pandas import option_context
    >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0 or len(args) < 2:
            raise ValueError('Need to invoke as option_context(pat, val, [(pat, val), ...]).')
        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self) -> None:
        self.undo = [(pat, _get_option(pat)) for pat, val in self.ops]
        for pat, val in self.ops:
            _set_option(pat, val, silent=True)

    def __exit__(self, *args) -> None:
        if self.undo:
            for pat, val in self.undo:
                _set_option(pat, val, silent=True)