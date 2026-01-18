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
class BaseCompletionType:
    """Describes different completion types"""

    def __init__(self, shown_before_tab: bool=True, mode: AutocompleteModes=AutocompleteModes.SIMPLE) -> None:
        self._shown_before_tab = shown_before_tab
        self.method_match = _MODES_MAP[mode]

    @abc.abstractmethod
    def matches(self, cursor_offset: int, line: str, **kwargs: Any) -> Optional[Set[str]]:
        """Returns a list of possible matches given a line and cursor, or None
        if this completion type isn't applicable.

        ie, import completion doesn't make sense if there cursor isn't after
        an import or from statement, so it ought to return None.

        Completion types are used to:
            * `locate(cur, line)` their initial target word to replace given a
              line and cursor
            * find `matches(cur, line)` that might replace that word
            * `format(match)` matches to be displayed to the user
            * determine whether suggestions should be `shown_before_tab`
            * `substitute(cur, line, match)` in a match for what's found with
              `target`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        """Returns a Linepart namedtuple instance or None given cursor and line

        A Linepart namedtuple contains a start, stop, and word. None is
        returned if no target for this type of completion is found under
        the cursor."""
        raise NotImplementedError

    def format(self, word: str) -> str:
        return word

    def substitute(self, cursor_offset: int, line: str, match: str) -> Tuple[int, str]:
        """Returns a cursor offset and line with match swapped in"""
        lpart = self.locate(cursor_offset, line)
        assert lpart
        offset = lpart.start + len(match)
        changed_line = line[:lpart.start] + match + line[lpart.stop:]
        return (offset, changed_line)

    @property
    def shown_before_tab(self) -> bool:
        """Whether suggestions should be shown before the user hits tab, or only
        once that has happened."""
        return self._shown_before_tab