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
class __compat__(__config_flags):
    """
    A cross-version compatibility configuration for pyparsing features that will be
    released in a future version. By setting values in this configuration to True,
    those features can be enabled in prior versions for compatibility development
    and testing.

    - ``collect_all_And_tokens`` - flag to enable fix for Issue #63 that fixes erroneous grouping
      of results names when an :class:`And` expression is nested within an :class:`Or` or :class:`MatchFirst`;
      maintained for compatibility, but setting to ``False`` no longer restores pre-2.3.1
      behavior
    """
    _type_desc = 'compatibility'
    collect_all_And_tokens = True
    _all_names = [__ for __ in locals() if not __.startswith('_')]
    _fixed_names = '\n        collect_all_And_tokens\n        '.split()