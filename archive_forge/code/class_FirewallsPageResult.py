from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
class FirewallsPageResult(NamedTuple):
    firewalls: list[BoundFirewall]
    meta: Meta | None