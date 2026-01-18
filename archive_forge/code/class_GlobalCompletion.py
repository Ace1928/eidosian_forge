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
class GlobalCompletion(BaseCompletionType):

    def matches(self, cursor_offset: int, line: str, *, locals_: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Optional[Set[str]]:
        """Compute matches when text is a simple name.
        Return a list of all keywords, built-in functions and names currently
        defined in self.namespace that match.
        """
        if locals_ is None:
            return None
        r = self.locate(cursor_offset, line)
        if r is None:
            return None
        n = len(r.word)
        matches = {word for word in KEYWORDS if self.method_match(word, n, r.word)}
        for nspace in (builtins.__dict__, locals_):
            for word, val in nspace.items():
                if word is None:
                    continue
                if self.method_match(word, n, r.word) and word != '__builtins__':
                    matches.add(_callable_postfix(val, word))
        return matches if matches else None

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_single_word(cursor_offset, line)