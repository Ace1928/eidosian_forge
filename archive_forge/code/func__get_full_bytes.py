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
def _get_full_bytes(sequence):
    """Return a byte string from a sequence of lines of bytes.

        This method detects if the sequence's lines are newline-terminated, and
        constructs the byte string appropriately.
        """
    sequence_iter = iter(sequence)
    try:
        first_line = next(sequence_iter)
    except StopIteration:
        return b''
    join_str = b'\n'
    if first_line.endswith(b'\n'):
        join_str = b''
    return first_line + join_str + join_str.join(sequence_iter)