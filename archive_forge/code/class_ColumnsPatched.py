from __future__ import annotations
import logging # isort:skip
from typing import (
class ColumnsPatched(TypedDict):
    kind: Literal['ColumnsPatched']
    model: Ref
    attr: str
    patches: Patches