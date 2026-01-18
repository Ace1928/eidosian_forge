from __future__ import annotations
import io
import json
import os
import typing as t
from .encoding import (
class SortedSetEncoder(json.JSONEncoder):
    """Encode sets as sorted lists."""

    def default(self, o: t.Any) -> t.Any:
        """Return a serialized version of the `o` object."""
        if isinstance(o, set):
            return sorted(o)
        return json.JSONEncoder.default(self, o)