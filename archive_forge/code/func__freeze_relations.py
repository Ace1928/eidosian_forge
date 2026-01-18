from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def _freeze_relations(self, relations: list[_Selector]) -> ct.SelectorList:
    """Freeze relation."""
    if relations:
        sel = relations[0]
        sel.relations.extend(relations[1:])
        return ct.SelectorList([sel.freeze()])
    else:
        return ct.SelectorList()