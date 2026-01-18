from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
@dataclasses.dataclass(frozen=True)
class VmResource(Resource):
    """Details needed to request a VM from Ansible Core CI."""
    platform: str
    version: str
    architecture: str
    provider: str
    tag: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return the resource as a tuple of platform, version, architecture and provider."""
        return (self.platform, self.version, self.architecture, self.provider)

    def get_label(self) -> str:
        """Return a user-friendly label for this resource."""
        return f'{self.platform} {self.version} ({self.architecture}) [{self.tag}] @{self.provider}'

    @property
    def persist(self) -> bool:
        """True if the resource is persistent, otherwise false."""
        return True