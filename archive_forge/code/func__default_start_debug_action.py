from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
def _default_start_debug_action(instring: str, loc: int, expr: 'ParserElement', cache_hit: bool=False):
    cache_hit_str = '*' if cache_hit else ''
    print(f'{cache_hit_str}Match {expr} at loc {loc}({lineno(loc, instring)},{col(loc, instring)})\n  {line(loc, instring)}\n  {' ' * (col(loc, instring) - 1)}^')