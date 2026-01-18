from __future__ import annotations
import typing as T
from typing_extensions import Literal, TypedDict, Required
class LibTarget(_BaseBuildTarget, total=False):
    name: str