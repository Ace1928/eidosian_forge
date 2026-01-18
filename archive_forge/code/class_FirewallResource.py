from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ..core import BaseDomain
class FirewallResource(BaseDomain):
    """Firewall Used By Domain

    :param type: str
           Type of resource referenced
    :param server: Optional[Server]
           Server the Firewall is applied to
    :param label_selector: Optional[FirewallResourceLabelSelector]
           Label Selector for Servers the Firewall should be applied to
    :param applied_to_resources: (read-only) List of effective resources the firewall is
           applied to.
    """
    __slots__ = ('type', 'server', 'label_selector', 'applied_to_resources')
    TYPE_SERVER = 'server'
    'Firewall Used By Type Server'
    TYPE_LABEL_SELECTOR = 'label_selector'
    'Firewall Used By Type label_selector'

    def __init__(self, type: str, server: Server | BoundServer | None=None, label_selector: FirewallResourceLabelSelector | None=None, applied_to_resources: list[FirewallResourceAppliedToResources] | None=None):
        self.type = type
        self.server = server
        self.label_selector = label_selector
        self.applied_to_resources = applied_to_resources

    def to_payload(self) -> dict[str, Any]:
        """
        Generates the request payload from this domain object.
        """
        payload: dict[str, Any] = {'type': self.type}
        if self.server is not None:
            payload['server'] = {'id': self.server.id}
        if self.label_selector is not None:
            payload['label_selector'] = {'selector': self.label_selector.selector}
        return payload