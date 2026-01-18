from __future__ import annotations
import sys
import types
from typing import (
class LifespanStartupEvent(TypedDict):
    type: Literal['lifespan.startup']