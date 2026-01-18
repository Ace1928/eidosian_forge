from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
def _or(obj: Model, selectors: Iterable[SelectorType]) -> bool:
    return any((match(obj, selector) for selector in selectors))