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
class __diag__(__config_flags):
    _type_desc = 'diagnostic'
    warn_multiple_tokens_in_named_alternation = False
    warn_ungrouped_named_tokens_in_collection = False
    warn_name_set_on_empty_Forward = False
    warn_on_parse_using_empty_Forward = False
    warn_on_assignment_to_Forward = False
    warn_on_multiple_string_args_to_oneof = False
    warn_on_match_first_with_lshift_operator = False
    enable_debug_on_named_expressions = False
    _all_names = [__ for __ in locals() if not __.startswith('_')]
    _warning_names = [name for name in _all_names if name.startswith('warn')]
    _debug_names = [name for name in _all_names if name.startswith('enable_debug')]

    @classmethod
    def enable_all_warnings(cls) -> None:
        for name in cls._warning_names:
            cls.enable(name)