from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
class BoundFirewall(BoundModelBase, Firewall):
    _client: FirewallsClient
    model = Firewall

    def __init__(self, client: FirewallsClient, data: dict, complete: bool=True):
        rules = data.get('rules', [])
        if rules:
            rules = [FirewallRule(direction=rule['direction'], source_ips=rule['source_ips'], destination_ips=rule['destination_ips'], protocol=rule['protocol'], port=rule['port'], description=rule['description']) for rule in rules]
            data['rules'] = rules
        applied_to = data.get('applied_to', [])
        if applied_to:
            from ..servers import BoundServer
            data_applied_to = []
            for firewall_resource in applied_to:
                applied_to_resources = None
                if firewall_resource.get('applied_to_resources'):
                    applied_to_resources = [FirewallResourceAppliedToResources(type=resource['type'], server=BoundServer(client._client.servers, resource.get('server'), complete=False) if resource.get('server') is not None else None) for resource in firewall_resource.get('applied_to_resources')]
                if firewall_resource['type'] == FirewallResource.TYPE_SERVER:
                    data_applied_to.append(FirewallResource(type=firewall_resource['type'], server=BoundServer(client._client.servers, firewall_resource['server'], complete=False), applied_to_resources=applied_to_resources))
                elif firewall_resource['type'] == FirewallResource.TYPE_LABEL_SELECTOR:
                    data_applied_to.append(FirewallResource(type=firewall_resource['type'], label_selector=FirewallResourceLabelSelector(selector=firewall_resource['label_selector']['selector']), applied_to_resources=applied_to_resources))
            data['applied_to'] = data_applied_to
        super().__init__(client, data, complete)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a Firewall.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :param page: int (optional)
               Specifies the page to fetch
        :param per_page: int (optional)
               Specifies how many results are returned by page
        :return: (List[:class:`BoundAction <hcloud.actions.client.BoundAction>`], :class:`Meta <hcloud.core.domain.Meta>`)
        """
        return self._client.get_actions_list(self, status, sort, page, per_page)

    def get_actions(self, status: list[str] | None=None, sort: list[str] | None=None) -> list[BoundAction]:
        """Returns all action objects for a Firewall.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`

        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundFirewall:
        """Updates the name or labels of a Firewall.

        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str (optional)
               New Name to set
        :return: :class:`BoundFirewall <hcloud.firewalls.client.BoundFirewall>`
        """
        return self._client.update(self, labels, name)

    def delete(self) -> bool:
        """Deletes a Firewall.

        :return: boolean
        """
        return self._client.delete(self)

    def set_rules(self, rules: list[FirewallRule]) -> list[BoundAction]:
        """Sets the rules of a Firewall. All existing rules will be overwritten. Pass an empty rules array to remove all rules.
        :param rules: List[:class:`FirewallRule <hcloud.firewalls.domain.FirewallRule>`]
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.set_rules(self, rules)

    def apply_to_resources(self, resources: list[FirewallResource]) -> list[BoundAction]:
        """Applies one Firewall to multiple resources.
        :param resources: List[:class:`FirewallResource <hcloud.firewalls.domain.FirewallResource>`]
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.apply_to_resources(self, resources)

    def remove_from_resources(self, resources: list[FirewallResource]) -> list[BoundAction]:
        """Removes one Firewall from multiple resources.
        :param resources: List[:class:`FirewallResource <hcloud.firewalls.domain.FirewallResource>`]
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.remove_from_resources(self, resources)