from __future__ import annotations
from functools import wraps
import inspect
from textwrap import dedent
from typing import (
import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _format_argument_list(allow_args: list[str]) -> str:
    """
    Convert the allow_args argument (either string or integer) of
    `deprecate_nonkeyword_arguments` function to a string describing
    it to be inserted into warning message.

    Parameters
    ----------
    allowed_args : list, tuple or int
        The `allowed_args` argument for `deprecate_nonkeyword_arguments`,
        but None value is not allowed.

    Returns
    -------
    str
        The substring describing the argument list in best way to be
        inserted to the warning message.

    Examples
    --------
    `format_argument_list([])` -> ''
    `format_argument_list(['a'])` -> "except for the arguments 'a'"
    `format_argument_list(['a', 'b'])` -> "except for the arguments 'a' and 'b'"
    `format_argument_list(['a', 'b', 'c'])` ->
        "except for the arguments 'a', 'b' and 'c'"
    """
    if 'self' in allow_args:
        allow_args.remove('self')
    if not allow_args:
        return ''
    elif len(allow_args) == 1:
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[-1]
        args = ', '.join(["'" + x + "'" for x in allow_args[:-1]])
        return f" except for the arguments {args} and '{last}'"