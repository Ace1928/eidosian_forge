from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class LoadBalancerTargetLabelSelector(BaseDomain):
    """LoadBalancerTargetLabelSelector Domain

    :param selector: str Target label selector
    """

    def __init__(self, selector: str | None=None):
        self.selector = selector