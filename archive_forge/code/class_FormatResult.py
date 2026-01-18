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
class FormatResult:
    """Possible exit codes."""
    ok = 0
    error = 1
    interrupted = 2
    check_failed = 3