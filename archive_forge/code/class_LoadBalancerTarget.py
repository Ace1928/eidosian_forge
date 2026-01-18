from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class LoadBalancerTarget(BaseDomain):
    """LoadBalancerTarget Domain

    :param type: str
            Type of the resource, can be server or label_selector
    :param server: Server
            Target server
    :param label_selector: LoadBalancerTargetLabelSelector
            Target label selector
    :param ip: LoadBalancerTargetIP
            Target IP
    :param use_private_ip: bool
            use the private IP instead of primary public IP
    :param health_status: list
            List of health statuses of the services on this target. Only present for target types "server" and "ip".
    """

    def __init__(self, type: str | None=None, server: BoundServer | None=None, label_selector: LoadBalancerTargetLabelSelector | None=None, ip: LoadBalancerTargetIP | None=None, use_private_ip: bool | None=None, health_status: list[LoadBalancerTargetHealthStatus] | None=None):
        self.type = type
        self.server = server
        self.label_selector = label_selector
        self.ip = ip
        self.use_private_ip = use_private_ip
        self.health_status = health_status

    def to_payload(self) -> dict[str, Any]:
        """
        Generates the request payload from this domain object.
        """
        payload: dict[str, Any] = {'type': self.type}
        if self.use_private_ip is not None:
            payload['use_private_ip'] = self.use_private_ip
        if self.type == 'server':
            if self.server is None:
                raise ValueError(f'server is not defined in target {self!r}')
            payload['server'] = {'id': self.server.id}
        elif self.type == 'label_selector':
            if self.label_selector is None:
                raise ValueError(f'label_selector is not defined in target {self!r}')
            payload['label_selector'] = {'selector': self.label_selector.selector}
        elif self.type == 'ip':
            if self.ip is None:
                raise ValueError(f'ip is not defined in target {self!r}')
            payload['ip'] = {'ip': self.ip.ip}
        return payload