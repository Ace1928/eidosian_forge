from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
class LayoutMessages:
    """Messages generated during layout creation that should be deferred for later display."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []