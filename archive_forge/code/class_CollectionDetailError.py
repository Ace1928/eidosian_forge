from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
class CollectionDetailError(ApplicationError):
    """An error occurred retrieving collection detail."""

    def __init__(self, reason: str) -> None:
        super().__init__('Error collecting collection detail: %s' % reason)
        self.reason = reason