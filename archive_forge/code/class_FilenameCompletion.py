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
class FilenameCompletion(BaseCompletionType):

    def __init__(self, mode: AutocompleteModes=AutocompleteModes.SIMPLE):
        super().__init__(False, mode)

    def matches(self, cursor_offset: int, line: str, **kwargs: Any) -> Optional[Set[str]]:
        cs = lineparts.current_string(cursor_offset, line)
        if cs is None:
            return None
        matches = set()
        username = cs.word.split(os.path.sep, 1)[0]
        user_dir = os.path.expanduser(username)
        for filename in _safe_glob(os.path.expanduser(cs.word)):
            if os.path.isdir(filename):
                filename += os.path.sep
            if cs.word.startswith('~'):
                filename = username + filename[len(user_dir):]
            matches.add(filename)
        return matches

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_string(cursor_offset, line)

    def format(self, filename: str) -> str:
        if os.sep in filename[:-1]:
            return filename[filename.rindex(os.sep, 0, -1) + 1:]
        else:
            return filename