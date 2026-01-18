from __future__ import annotations
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
class CoverageAnalyzeTargetsCombineConfig(CoverageAnalyzeTargetsConfig):
    """Configuration for the `coverage analyze targets combine` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.input_files: list[str] = args.input_file
        self.output_file: str = args.output_file