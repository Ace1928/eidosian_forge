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
def _should_enable_warnings(cmd_line_warn_options: typing.Iterable[str], warn_env_var: typing.Optional[str]) -> bool:
    enable = bool(warn_env_var)
    for warn_opt in cmd_line_warn_options:
        w_action, w_message, w_category, w_module, w_line = (warn_opt + '::::').split(':')[:5]
        if not w_action.lower().startswith('i') and (not (w_message or w_category or w_module) or w_module == 'pyparsing'):
            enable = True
        elif w_action.lower().startswith('i') and w_module in ('pyparsing', ''):
            enable = False
    return enable