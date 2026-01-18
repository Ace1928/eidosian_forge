from __future__ import annotations
import logging # isort:skip
from typing import (
class TitleChanged(TypedDict):
    kind: Literal['TitleChanged']
    title: str