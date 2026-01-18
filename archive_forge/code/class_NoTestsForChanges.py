from __future__ import annotations
import typing as t
from .io import (
from .util import (
from .ci import (
from .classification import (
from .config import (
from .metadata import (
from .provisioning import (
class NoTestsForChanges(ApplicationWarning):
    """Exception when changes detected, but no tests trigger as a result."""

    def __init__(self) -> None:
        super().__init__('No tests found for detected changes.')