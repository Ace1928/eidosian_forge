from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
def _get_changes(self, args: CommonConfig) -> AzurePipelinesChanges:
    """Return an AzurePipelinesChanges instance, which will be created on first use."""
    if not self._changes:
        self._changes = AzurePipelinesChanges(args)
    return self._changes