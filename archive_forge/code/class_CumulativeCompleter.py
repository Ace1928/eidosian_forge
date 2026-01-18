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
class CumulativeCompleter(BaseCompletionType):
    """Returns combined matches from several completers"""

    def __init__(self, completers: Sequence[BaseCompletionType], mode: AutocompleteModes=AutocompleteModes.SIMPLE) -> None:
        if not completers:
            raise ValueError('CumulativeCompleter requires at least one completer')
        self._completers: Sequence[BaseCompletionType] = completers
        super().__init__(True, mode)

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        for completer in self._completers:
            return_value = completer.locate(cursor_offset, line)
            if return_value is not None:
                return return_value
        return None

    def format(self, word: str) -> str:
        return self._completers[0].format(word)

    def matches(self, cursor_offset: int, line: str, **kwargs: Any) -> Optional[Set[str]]:
        return_value = None
        all_matches = set()
        for completer in self._completers:
            matches = completer.matches(cursor_offset=cursor_offset, line=line, **kwargs)
            if matches is not None:
                all_matches.update(matches)
                return_value = all_matches
        return return_value