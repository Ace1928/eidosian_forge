from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Network, NetworkRoute, NetworkSubnet
class BoundNetwork(BoundModelBase, Network):
    _client: NetworksClient
    model = Network

    def __init__(self, client: NetworksClient, data: dict, complete: bool=True):
        subnets = data.get('subnets', [])
        if subnets is not None:
            subnets = [NetworkSubnet.from_dict(subnet) for subnet in subnets]
            data['subnets'] = subnets
        routes = data.get('routes', [])
        if routes is not None:
            routes = [NetworkRoute.from_dict(route) for route in routes]
            data['routes'] = routes
        from ..servers import BoundServer
        servers = data.get('servers', [])
        if servers is not None:
            servers = [BoundServer(client._client.servers, {'id': server}, complete=False) for server in servers]
            data['servers'] = servers
        super().__init__(client, data, complete)

    def update(self, name: str | None=None, expose_routes_to_vswitch: bool | None=None, labels: dict[str, str] | None=None) -> BoundNetwork:
        """Updates a network. You can update a network’s name and a networks’s labels.

        :param name: str (optional)
               New name to set
        :param expose_routes_to_vswitch: Optional[bool]
                Indicates if the routes from this network should be exposed to the vSwitch connection.
                The exposing only takes effect if a vSwitch connection is active.
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>`
        """
        return self._client.update(self, name=name, expose_routes_to_vswitch=expose_routes_to_vswitch, labels=labels)

    def delete(self) -> bool:
        """Deletes a network.

        :return: boolean
        """
        return self._client.delete(self)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a network.

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
        """Returns all action objects for a network.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def add_subnet(self, subnet: NetworkSubnet) -> BoundAction:
        """Adds a subnet entry to a network.

        :param subnet: :class:`NetworkSubnet <hcloud.networks.domain.NetworkSubnet>`
                       The NetworkSubnet you want to add to the Network
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.add_subnet(self, subnet=subnet)

    def delete_subnet(self, subnet: NetworkSubnet) -> BoundAction:
        """Removes a subnet entry from a network

        :param subnet: :class:`NetworkSubnet <hcloud.networks.domain.NetworkSubnet>`
                       The NetworkSubnet you want to remove from the Network
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.delete_subnet(self, subnet=subnet)

    def add_route(self, route: NetworkRoute) -> BoundAction:
        """Adds a route entry to a network.

        :param route: :class:`NetworkRoute <hcloud.networks.domain.NetworkRoute>`
                    The NetworkRoute you want to add to the Network
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.add_route(self, route=route)

    def delete_route(self, route: NetworkRoute) -> BoundAction:
        """Removes a route entry to a network.

        :param route: :class:`NetworkRoute <hcloud.networks.domain.NetworkRoute>`
                    The NetworkRoute you want to remove from the Network
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.delete_route(self, route=route)

    def change_ip_range(self, ip_range: str) -> BoundAction:
        """Changes the IP range of a network.

        :param ip_range: str
                    The new prefix for the whole network.
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_ip_range(self, ip_range=ip_range)

    def change_protection(self, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of a network.

        :param delete: boolean
               If True, prevents the network from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete=delete)