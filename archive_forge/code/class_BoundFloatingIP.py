from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from .domain import CreateFloatingIPResponse, FloatingIP
class BoundFloatingIP(BoundModelBase, FloatingIP):
    _client: FloatingIPsClient
    model = FloatingIP

    def __init__(self, client: FloatingIPsClient, data: dict, complete: bool=True):
        from ..servers import BoundServer
        server = data.get('server')
        if server is not None:
            data['server'] = BoundServer(client._client.servers, {'id': server}, complete=False)
        home_location = data.get('home_location')
        if home_location is not None:
            data['home_location'] = BoundLocation(client._client.locations, home_location)
        super().__init__(client, data, complete)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a Floating IP.

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
        """Returns all action objects for a Floating IP.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def update(self, description: str | None=None, labels: dict[str, str] | None=None, name: str | None=None) -> BoundFloatingIP:
        """Updates the description or labels of a Floating IP.

        :param description: str (optional)
               New Description to set
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str (optional)
               New Name to set
        :return: :class:`BoundFloatingIP <hcloud.floating_ips.client.BoundFloatingIP>`
        """
        return self._client.update(self, description, labels, name)

    def delete(self) -> bool:
        """Deletes a Floating IP. If it is currently assigned to a server it will automatically get unassigned.

        :return: boolean
        """
        return self._client.delete(self)

    def change_protection(self, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of the Floating IP.

        :param delete: boolean
               If true, prevents the Floating IP from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete)

    def assign(self, server: Server | BoundServer) -> BoundAction:
        """Assigns a Floating IP to a server.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
               Server the Floating IP shall be assigned to
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.assign(self, server)

    def unassign(self) -> BoundAction:
        """Unassigns a Floating IP, resulting in it being unreachable. You may assign it to a server again at a later time.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.unassign(self)

    def change_dns_ptr(self, ip: str, dns_ptr: str) -> BoundAction:
        """Changes the hostname that will appear when getting the hostname belonging to this Floating IP.

        :param ip: str
               The IP address for which to set the reverse DNS entry
        :param dns_ptr: str
               Hostname to set as a reverse DNS PTR entry, will reset to original default value if `None`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_dns_ptr(self, ip, dns_ptr)