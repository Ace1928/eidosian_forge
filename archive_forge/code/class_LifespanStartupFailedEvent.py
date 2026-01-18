from __future__ import annotations
import sys
import types
from typing import (
class LifespanStartupFailedEvent(TypedDict):
    type: Literal['lifespan.startup.failed']
    message: str