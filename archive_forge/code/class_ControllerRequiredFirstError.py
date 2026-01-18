from __future__ import annotations
import abc
import typing as t
from ..argparsing.parsers import (
class ControllerRequiredFirstError(CompletionError):
    """Exception raised when controller and target options are specified out-of-order."""

    def __init__(self) -> None:
        super().__init__('The `--controller` option must be specified before `--target` option(s).')