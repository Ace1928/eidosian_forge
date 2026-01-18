import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def _do_wrap_field(field_name, field_body, indentation, wrap_length):
    """Wrap complete field at wrap_length characters.

    Parameters
    ----------
    field_name : str
        The name text of the field.
    field_body : str
        The body text of the field.
    indentation : str
        The string to use for indentation of the first line in the field.
    wrap_length : int
        The number of characters at which to wrap the field.

    Returns
    -------
    _wrapped_field : str
        The field wrapped at wrap_length characters.
    """
    if len(indentation) > DEFAULT_INDENT:
        _subsequent = indentation + int(0.5 * len(indentation)) * ' '
    else:
        _subsequent = 2 * indentation
    _wrapped_field = textwrap.wrap(textwrap.dedent(f'{field_name}{field_body}'), width=wrap_length, initial_indent=indentation, subsequent_indent=_subsequent)
    for _idx, _field in enumerate(_wrapped_field):
        _indent = indentation if _idx == 0 else _subsequent
        _wrapped_field[_idx] = f'{_indent}{re.sub(' +', ' ', _field.strip())}'
    return _wrapped_field