from __future__ import annotations
import os
import typing as t
from ...io import (
from ...util import (
from ...data import (
from ...provisioning import (
from .combine import (
from . import (
class CoverageReportConfig(CoverageCombineConfig):
    """Configuration for the coverage report command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.show_missing: bool = args.show_missing
        self.include: str = args.include
        self.omit: str = args.omit