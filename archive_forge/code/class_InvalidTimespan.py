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
class InvalidTimespan(Exception):
    """
    Raised when a string cannot be parsed into a timespan.

    For example:

    >>> from humanfriendly import parse_timespan
    >>> parse_timespan('1 age')
    Traceback (most recent call last):
      File "humanfriendly/__init__.py", line 419, in parse_timespan
        raise InvalidTimespan(format(msg, timespan, tokens))
    humanfriendly.InvalidTimespan: Failed to parse timespan! (input '1 age' was tokenized as [1, 'age'])
    """