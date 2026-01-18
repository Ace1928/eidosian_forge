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
class ParameterNameCompletion(BaseCompletionType):

    def matches(self, cursor_offset: int, line: str, *, funcprops: Optional[inspection.FuncProps]=None, **kwargs: Any) -> Optional[Set[str]]:
        if funcprops is None:
            return None
        r = self.locate(cursor_offset, line)
        if r is None:
            return None
        matches = {f'{name}=' for name in funcprops.argspec.args if isinstance(name, str) and name.startswith(r.word)}
        matches.update((f'{name}=' for name in funcprops.argspec.kwonly if name.startswith(r.word)))
        return matches if matches else None

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_word(cursor_offset, line)