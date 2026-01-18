from __future__ import annotations
import logging # isort:skip
from typing import (
class ModelChanged(TypedDict):
    kind: Literal['ModelChanged']
    model: Ref
    attr: str
    new: Any