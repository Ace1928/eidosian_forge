from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....executor import (
from .....provisioning import (
from . import (
from . import (
class CoverageAnalyzeTargetsMissingConfig(CoverageAnalyzeTargetsConfig):
    """Configuration for the `coverage analyze targets missing` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.from_file: str = args.from_file
        self.to_file: str = args.to_file
        self.output_file: str = args.output_file
        self.only_gaps: bool = args.only_gaps
        self.only_exists: bool = args.only_exists