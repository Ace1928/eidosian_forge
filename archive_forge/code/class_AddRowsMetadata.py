from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, TypedDict
@dataclass
class AddRowsMetadata:
    """Metadata needed by add_rows on native charts."""
    last_index: Hashable | None
    columns: PrepDataColumns