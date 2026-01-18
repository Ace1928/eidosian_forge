from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from warnings import warn
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Iso
class IsosPageResult(NamedTuple):
    isos: list[BoundIso]
    meta: Meta | None