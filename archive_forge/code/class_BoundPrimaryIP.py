from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePrimaryIPResponse, PrimaryIP
class BoundPrimaryIP(BoundModelBase, PrimaryIP):
    _client: PrimaryIPsClient
    model = PrimaryIP

    def __init__(self, client: PrimaryIPsClient, data: dict, complete: bool=True):
        from ..datacenters import BoundDatacenter
        datacenter = data.get('datacenter', {})
        if datacenter:
            data['datacenter'] = BoundDatacenter(client._client.datacenters, datacenter)
        super().__init__(client, data, complete)

    def update(self, auto_delete: bool | None=None, labels: dict[str, str] | None=None, name: str | None=None) -> BoundPrimaryIP:
        """Updates the description or labels of a Primary IP.

        :param auto_delete: bool (optional)
               Auto delete IP when assignee gets deleted
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str (optional)
               New Name to set
        :return: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`
        """
        return self._client.update(self, auto_delete=auto_delete, labels=labels, name=name)

    def delete(self) -> bool:
        """Deletes a Primary IP. If it is currently assigned to a server it will automatically get unassigned.

        :return: boolean
        """
        return self._client.delete(self)

    def change_protection(self, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of the Primary IP.

        :param delete: boolean
               If true, prevents the Primary IP from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete)

    def assign(self, assignee_id: int, assignee_type: str) -> BoundAction:
        """Assigns a Primary IP to a assignee.

        :param assignee_id: int`
               Id of an assignee the Primary IP shall be assigned to
        :param assignee_type: string`
               Assignee type (e.g server) the Primary IP shall be assigned to
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.assign(self, assignee_id, assignee_type)

    def unassign(self) -> BoundAction:
        """Unassigns a Primary IP, resulting in it being unreachable. You may assign it to a server again at a later time.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.unassign(self)

    def change_dns_ptr(self, ip: str, dns_ptr: str) -> BoundAction:
        """Changes the hostname that will appear when getting the hostname belonging to this Primary IP.

        :param ip: str
               The IP address for which to set the reverse DNS entry
        :param dns_ptr: str
               Hostname to set as a reverse DNS PTR entry, will reset to original default value if `None`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_dns_ptr(self, ip, dns_ptr)