from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
class AutoName(Enum):
    """
    This is used for creating Enum classes where `auto()` is the string form
    of the corresponding enum's identifier (e.g. FOO.value results in "FOO").

    Reference: https://docs.python.org/3/howto/enum.html#using-automatic-values
    """

    def _generate_next_value_(name, _start, _count, _last_values):
        return name