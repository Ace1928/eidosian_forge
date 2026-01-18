from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import LoadBalancerType
class BoundLoadBalancerType(BoundModelBase, LoadBalancerType):
    _client: LoadBalancerTypesClient
    model = LoadBalancerType