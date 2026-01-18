from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePlacementGroupResponse, PlacementGroup
class PlacementGroupsPageResult(NamedTuple):
    placement_groups: list[BoundPlacementGroup]
    meta: Meta | None