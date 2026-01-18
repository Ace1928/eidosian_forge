from __future__ import annotations
import typing as t
from .io import (
from .util import (
from .ci import (
from .classification import (
from .config import (
from .metadata import (
from .provisioning import (
class NoChangesDetected(ApplicationWarning):
    """Exception when change detection was performed, but no changes were found."""

    def __init__(self) -> None:
        super().__init__('No changes detected.')