import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize
class InvalidLength(Exception):
    """
    Raised when a string cannot be parsed into a length.

    For example:

    >>> from humanfriendly import parse_length
    >>> parse_length('5 Z')
    Traceback (most recent call last):
      File "humanfriendly/__init__.py", line 267, in parse_length
        raise InvalidLength(format(msg, length, tokens))
    humanfriendly.InvalidLength: Failed to parse length! (input '5 Z' was tokenized as [5, 'Z'])
    """