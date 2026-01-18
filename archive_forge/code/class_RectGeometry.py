from __future__ import annotations
import logging # isort:skip
import os
from typing import (
class RectGeometry(TypedDict):
    type: Literal['rect']
    sx0: float
    sx1: float
    sy0: float
    sy1: float