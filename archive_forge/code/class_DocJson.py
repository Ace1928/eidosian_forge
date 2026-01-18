from __future__ import annotations
import logging # isort:skip
from typing import (
class DocJson(TypedDict):
    version: NotRequired[str]
    title: NotRequired[str]
    defs: NotRequired[list[ModelDef]]
    roots: list[ModelRep]
    callbacks: NotRequired[dict[str, list[ModelRep]]]