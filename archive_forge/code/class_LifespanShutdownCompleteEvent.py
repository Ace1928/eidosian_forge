from __future__ import annotations
import sys
import types
from typing import (
class LifespanShutdownCompleteEvent(TypedDict):
    type: Literal['lifespan.shutdown.complete']