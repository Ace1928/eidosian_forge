from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
def _classify(self, path: str) -> t.Optional[dict[str, str]]:
    """Return the classification for the given path."""
    if data_context().content.is_ansible:
        return self._classify_ansible(path)
    if data_context().content.collection:
        return self._classify_collection(path)
    return None