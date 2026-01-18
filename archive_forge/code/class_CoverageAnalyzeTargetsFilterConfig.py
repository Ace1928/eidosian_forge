from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
class CoverageAnalyzeTargetsFilterConfig(CoverageAnalyzeTargetsConfig):
    """Configuration for the `coverage analyze targets filter` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.input_file: str = args.input_file
        self.output_file: str = args.output_file
        self.include_targets: list[str] = args.include_targets
        self.exclude_targets: list[str] = args.exclude_targets
        self.include_path: t.Optional[str] = args.include_path
        self.exclude_path: t.Optional[str] = args.exclude_path