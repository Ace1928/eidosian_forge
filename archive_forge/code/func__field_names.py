from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Dict
from numpy import ndarray
@property
def _field_names(self) -> tuple[str, ...]:
    """Tuple of field names in any inheriting result dataclass."""
    return tuple((field.name for field in fields(self)))