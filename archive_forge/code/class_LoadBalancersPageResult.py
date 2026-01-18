from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..certificates import BoundCertificate
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..load_balancer_types import BoundLoadBalancerType
from ..locations import BoundLocation
from ..metrics import Metrics
from ..networks import BoundNetwork
from ..servers import BoundServer
from .domain import (
class LoadBalancersPageResult(NamedTuple):
    load_balancers: list[BoundLoadBalancer]
    meta: Meta | None