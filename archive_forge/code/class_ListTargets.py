from __future__ import annotations
import typing as t
from .io import (
from .util import (
from .ci import (
from .classification import (
from .config import (
from .metadata import (
from .provisioning import (
class ListTargets(Exception):
    """List integration test targets instead of executing them."""

    def __init__(self, target_names: list[str]) -> None:
        super().__init__()
        self.target_names = target_names