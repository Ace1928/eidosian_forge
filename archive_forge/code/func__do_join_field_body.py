import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def _do_join_field_body(text, field_idx, idx):
    """Join the filed body lines into a single line that can be wrapped.

    Parameters
    ----------
    text : str
        The docstring long description text that contains field lists.
    field_idx : list
        The list of tuples containing the found field list start and end position.

    Returns
    -------
    _field_body : str
        The field body collapsed into a single line.
    """
    try:
        _field_body = text[field_idx[idx][1]:field_idx[idx + 1][0]].strip()
    except IndexError:
        _field_body = text[field_idx[idx][1]:].strip()
    _field_body = ' '.join([_line.strip() for _line in _field_body.splitlines()]).strip()
    if not _field_body.startswith('`') and _field_body:
        _field_body = f' {_field_body}'
    if text[field_idx[idx][1]:field_idx[idx][1] + 2] == '\n\n':
        _field_body = '\n'
    return _field_body