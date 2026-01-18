from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
@dataclasses.dataclass(frozen=True)
class PipCommand:
    """Base class for pip commands."""

    def serialize(self) -> tuple[str, dict[str, t.Any]]:
        """Return a serialized representation of this command."""
        name = type(self).__name__[3:].lower()
        return (name, self.__dict__)