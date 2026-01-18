import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_format_code(self, source):
    """Return source code with docstrings formatted.

        Parameters
        ----------
        source: str
            The text from the source file.
        """
    try:
        _original_newline = self.encodor.do_find_newline(source.splitlines(True))
        _code = self._format_code(source)
        return _strings.normalize_line_endings(_code.splitlines(True), _original_newline)
    except (tokenize.TokenError, IndentationError):
        return source