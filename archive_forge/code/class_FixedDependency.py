from __future__ import annotations
import typing as T
from typing_extensions import Literal, TypedDict, Required
class FixedDependency(TypedDict, total=False):
    """An entry in the *dependencies sections, fixed up."""
    version: T.List[str]
    registry: str
    git: str
    branch: str
    rev: str
    path: str
    optional: bool
    package: str
    default_features: bool
    features: T.List[str]