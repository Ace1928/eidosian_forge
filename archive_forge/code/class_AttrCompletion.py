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
class AttrCompletion(BaseCompletionType):
    attr_matches_re = LazyReCompile('(\\w+(\\.\\w+)*)\\.(\\w*)')

    def matches(self, cursor_offset: int, line: str, *, locals_: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Optional[Set[str]]:
        r = self.locate(cursor_offset, line)
        if r is None:
            return None
        if locals_ is None:
            locals_ = __main__.__dict__
        assert '.' in r.word
        i = r.word.rfind('[') + 1
        methodtext = r.word[i:]
        matches = {''.join([r.word[:i], m]) for m in self.attr_matches(methodtext, locals_)}
        return {m for m in matches if _few_enough_underscores(r.word.split('.')[-1], m.split('.')[-1])}

    def locate(self, cursor_offset: int, line: str) -> Optional[LinePart]:
        return lineparts.current_dotted_attribute(cursor_offset, line)

    def format(self, word: str) -> str:
        return _after_last_dot(word)

    def attr_matches(self, text: str, namespace: Dict[str, Any]) -> Iterator[str]:
        """Taken from rlcompleter.py and bent to my will."""
        m = self.attr_matches_re.match(text)
        if not m:
            return (_ for _ in ())
        expr, attr = m.group(1, 3)
        if expr.isdigit():
            return (_ for _ in ())
        try:
            obj = safe_eval(expr, namespace)
        except EvaluationError:
            return (_ for _ in ())
        return self.attr_lookup(obj, expr, attr)

    def attr_lookup(self, obj: Any, expr: str, attr: str) -> Iterator[str]:
        """Second half of attr_matches."""
        words = self.list_attributes(obj)
        if inspection.hasattr_safe(obj, '__class__'):
            words.append('__class__')
            klass = inspection.getattr_safe(obj, '__class__')
            words = words + rlcompleter.get_class_members(klass)
            if not isinstance(klass, abc.ABCMeta):
                try:
                    words.remove('__abstractmethods__')
                except ValueError:
                    pass
        n = len(attr)
        return (f'{expr}.{word}' for word in words if self.method_match(word, n, attr) and word != '__builtins__')

    def list_attributes(self, obj: Any) -> List[str]:
        with inspection.AttrCleaner(obj):
            return dir(obj)