from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Location
class LocationsPageResult(NamedTuple):
    locations: list[BoundLocation]
    meta: Meta | None