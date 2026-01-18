from __future__ import annotations
import sys
import types
from typing import (
class LifespanScope(TypedDict):
    type: Literal['lifespan']
    asgi: ASGIVersions
    state: NotRequired[dict[str, Any]]