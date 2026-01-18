import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
def _few_enough_underscores(current: str, match: str) -> bool:
    """Returns whether match should be shown based on current

    if current is _, True if match starts with 0 or 1 underscore
    if current is __, True regardless of match
    otherwise True if match does not start with any underscore
    """
    if current.startswith('__'):
        return True
    elif current.startswith('_') and (not match.startswith('__')):
        return True
    return not match.startswith('_')