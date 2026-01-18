from __future__ import annotations
import difflib
import typing as t
from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect
def _score_rule(rule: Rule) -> float:
    return sum([0.98 * difflib.SequenceMatcher(None, rule.endpoint, self.endpoint).ratio(), 0.01 * bool(set(self.values or ()).issubset(rule.arguments)), 0.01 * bool(rule.methods and self.method in rule.methods)])