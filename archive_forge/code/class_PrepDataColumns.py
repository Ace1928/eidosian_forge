from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, TypedDict
class PrepDataColumns(TypedDict):
    """Columns used for the prep_data step in Altair Arrow charts."""
    x_column: str | None
    y_column_list: list[str]
    color_column: str | None
    size_column: str | None