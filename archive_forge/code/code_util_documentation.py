from __future__ import annotations
import re
from typing import Any
Parse arguments from a stringified arguments list inside parentheses

    Parameters
    ----------
    args : list
        A list where it's size matches the expected number of parsed arguments
    line : str
        Stringified line of code with method arguments inside parentheses

    Returns
    -------
    list of strings
        Parsed arguments

    Example
    -------
    >>> line = 'foo(bar, baz, my(func, tion))'
    >>>
    >>> get_method_args_from_code(range(0, 3), line)
    ['bar', 'baz', 'my(func, tion)']

    