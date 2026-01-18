from __future__ import annotations
import typing as t
from .....io import (
from .....executor import (
from .....provisioning import (
from . import (
class CoverageAnalyzeTargetsExpandConfig(CoverageAnalyzeTargetsConfig):
    """Configuration for the `coverage analyze targets expand` command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args)
        self.input_file: str = args.input_file
        self.output_file: str = args.output_file