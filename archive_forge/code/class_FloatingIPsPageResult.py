from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from .domain import CreateFloatingIPResponse, FloatingIP
class FloatingIPsPageResult(NamedTuple):
    floating_ips: list[BoundFloatingIP]
    meta: Meta | None