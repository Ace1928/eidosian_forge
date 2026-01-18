import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@staticmethod
def _skip_useless_lines(sequence):
    """Yields only lines that do not begin with '#'.

        Also skips any blank lines at the beginning of the input.
        """
    at_beginning = True
    for line in sequence:
        if isinstance(line, str):
            line = line.encode()
        if line.startswith(b'#'):
            continue
        if at_beginning:
            if not line.rstrip(b'\r\n'):
                continue
            at_beginning = False
        yield line