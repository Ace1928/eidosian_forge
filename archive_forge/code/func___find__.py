from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
def __find__(self, name: str) -> int:
    """Find rule index by name"""
    for i, rule in enumerate(self.__rules__):
        if rule.name == name:
            return i
    return -1