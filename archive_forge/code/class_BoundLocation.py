from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Location
class BoundLocation(BoundModelBase, Location):
    _client: LocationsClient
    model = Location