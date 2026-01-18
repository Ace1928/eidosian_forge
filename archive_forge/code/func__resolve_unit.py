import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _resolve_unit(raw_match):
    if raw_match is None:
        return 'second'
    text = raw_match.lower()
    return _unit_lookup.get(text, text)