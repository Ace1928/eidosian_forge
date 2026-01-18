from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Network, NetworkRoute, NetworkSubnet
class NetworksPageResult(NamedTuple):
    networks: list[BoundNetwork]
    meta: Meta | None