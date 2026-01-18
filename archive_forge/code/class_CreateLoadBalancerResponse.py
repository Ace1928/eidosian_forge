from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class CreateLoadBalancerResponse(BaseDomain):
    """Create Load Balancer Response Domain

    :param load_balancer: :class:`BoundLoadBalancer <hcloud.load_balancers.client.BoundLoadBalancer>`
           The created Load Balancer
    :param action: :class:`BoundAction <hcloud.actions.client.BoundAction>`
           Shows the progress of the Load Balancer creation
    """
    __slots__ = ('load_balancer', 'action')

    def __init__(self, load_balancer: BoundLoadBalancer, action: BoundAction):
        self.load_balancer = load_balancer
        self.action = action