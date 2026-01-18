from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
class SpecialPseudoPattern(SelectorPattern):
    """Selector pattern."""

    def __init__(self, patterns: tuple[tuple[str, tuple[str, ...], str, type[SelectorPattern]], ...]) -> None:
        """Initialize."""
        self.patterns = {}
        for p in patterns:
            name = p[0]
            pattern = p[3](name, p[2])
            for pseudo in p[1]:
                self.patterns[pseudo] = pattern
        self.matched_name = None
        self.re_pseudo_name = re.compile(PAT_PSEUDO_CLASS_SPECIAL, re.I | re.X | re.U)

    def get_name(self) -> str:
        """Get name."""
        return '' if self.matched_name is None else self.matched_name.get_name()

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        """Match the selector."""
        pseudo = None
        m = self.re_pseudo_name.match(selector, index)
        if m:
            name = util.lower(css_unescape(m.group('name')))
            pattern = self.patterns.get(name)
            if pattern:
                pseudo = pattern.match(selector, index, flags)
                if pseudo:
                    self.matched_name = pattern
        return pseudo