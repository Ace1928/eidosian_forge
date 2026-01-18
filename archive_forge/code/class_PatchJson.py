from __future__ import annotations
import logging # isort:skip
from typing import (
class PatchJson(TypedDict):
    events: list[DocumentChanged]