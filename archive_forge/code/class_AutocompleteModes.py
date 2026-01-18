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
class AutocompleteModes(Enum):
    NONE = 'none'
    SIMPLE = 'simple'
    SUBSTRING = 'substring'
    FUZZY = 'fuzzy'

    @classmethod
    def from_string(cls, value: str) -> Optional['AutocompleteModes']:
        if value.upper() in cls.__members__:
            return cls.__members__[value.upper()]
        return None