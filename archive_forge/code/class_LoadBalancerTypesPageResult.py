from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import LoadBalancerType
class LoadBalancerTypesPageResult(NamedTuple):
    load_balancer_types: list[BoundLoadBalancerType]
    meta: Meta | None