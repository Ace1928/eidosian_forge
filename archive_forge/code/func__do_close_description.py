import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def _do_close_description(text: str, text_idx: int, indentation: str) -> List[str]:
    """Wrap any description following the last URL or field list.

    Parameters
    ----------
    text : str
        The docstring text.
    text_idx : int
        The index of the last URL or field list match.
    indentation : str
        The indentation string to use with docstrings.

    Returns
    -------
    _split_lines : str
        The text input split into individual lines.
    """
    _split_lines = []
    with contextlib.suppress(IndexError):
        _split_lines = (text[text_idx + 1:] if text[text_idx] == '\n' else text[text_idx:]).splitlines()
        for _idx, _line in enumerate(_split_lines):
            if _line not in ['', '\n', f'{indentation}']:
                _split_lines[_idx] = f'{indentation}{_line.strip()}'
    return _split_lines