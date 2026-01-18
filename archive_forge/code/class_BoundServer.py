from __future__ import annotations
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..datacenters import BoundDatacenter
from ..firewalls import BoundFirewall
from ..floating_ips import BoundFloatingIP
from ..images import BoundImage, CreateImageResponse
from ..isos import BoundIso
from ..metrics import Metrics
from ..placement_groups import BoundPlacementGroup
from ..primary_ips import BoundPrimaryIP
from ..server_types import BoundServerType
from ..volumes import BoundVolume
from .domain import (
class BoundServer(BoundModelBase, Server):
    _client: ServersClient
    model = Server

    def __init__(self, client: ServersClient, data: dict, complete: bool=True):
        datacenter = data.get('datacenter')
        if datacenter is not None:
            data['datacenter'] = BoundDatacenter(client._client.datacenters, datacenter)
        volumes = data.get('volumes', [])
        if volumes:
            volumes = [BoundVolume(client._client.volumes, {'id': volume}, complete=False) for volume in volumes]
            data['volumes'] = volumes
        image = data.get('image', None)
        if image is not None:
            data['image'] = BoundImage(client._client.images, image)
        iso = data.get('iso', None)
        if iso is not None:
            data['iso'] = BoundIso(client._client.isos, iso)
        server_type = data.get('server_type')
        if server_type is not None:
            data['server_type'] = BoundServerType(client._client.server_types, server_type)
        public_net = data.get('public_net')
        if public_net:
            ipv4_address = IPv4Address.from_dict(public_net['ipv4']) if public_net['ipv4'] is not None else None
            ipv4_primary_ip = BoundPrimaryIP(client._client.primary_ips, {'id': public_net['ipv4']['id']}, complete=False) if public_net['ipv4'] is not None else None
            ipv6_network = IPv6Network.from_dict(public_net['ipv6']) if public_net['ipv6'] is not None else None
            ipv6_primary_ip = BoundPrimaryIP(client._client.primary_ips, {'id': public_net['ipv6']['id']}, complete=False) if public_net['ipv6'] is not None else None
            floating_ips = [BoundFloatingIP(client._client.floating_ips, {'id': floating_ip}, complete=False) for floating_ip in public_net['floating_ips']]
            firewalls = [PublicNetworkFirewall(BoundFirewall(client._client.firewalls, {'id': firewall['id']}, complete=False), status=firewall['status']) for firewall in public_net.get('firewalls', [])]
            data['public_net'] = PublicNetwork(ipv4=ipv4_address, ipv6=ipv6_network, primary_ipv4=ipv4_primary_ip, primary_ipv6=ipv6_primary_ip, floating_ips=floating_ips, firewalls=firewalls)
        private_nets = data.get('private_net')
        if private_nets:
            from ..networks import BoundNetwork
            private_nets = [PrivateNet(network=BoundNetwork(client._client.networks, {'id': private_net['network']}, complete=False), ip=private_net['ip'], alias_ips=private_net['alias_ips'], mac_address=private_net['mac_address']) for private_net in private_nets]
            data['private_net'] = private_nets
        placement_group = data.get('placement_group')
        if placement_group:
            placement_group = BoundPlacementGroup(client._client.placement_groups, placement_group)
            data['placement_group'] = placement_group
        super().__init__(client, data, complete)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a server.

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
        """Returns all action objects for a server.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundServer:
        """Updates a server. You can update a server’s name and a server’s labels.

        :param name: str (optional)
               New name to set
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundServer <hcloud.servers.client.BoundServer>`
        """
        return self._client.update(self, name, labels)

    def get_metrics(self, type: MetricsType | list[MetricsType], start: datetime | str, end: datetime | str, step: float | None=None) -> GetMetricsResponse:
        """Get Metrics for a Server.

        :param server: The Server to get the metrics for.
        :param type: Type of metrics to get.
        :param start: Start of period to get Metrics for (in ISO-8601 format).
        :param end: End of period to get Metrics for (in ISO-8601 format).
        :param step: Resolution of results in seconds.
        """
        return self._client.get_metrics(self, type=type, start=start, end=end, step=step)

    def delete(self) -> BoundAction:
        """Deletes a server. This immediately removes the server from your account, and it is no longer accessible.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.delete(self)

    def power_off(self) -> BoundAction:
        """Cuts power to the server. This forcefully stops it without giving the server operating system time to gracefully stop

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.power_off(self)

    def power_on(self) -> BoundAction:
        """Starts a server by turning its power on.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.power_on(self)

    def reboot(self) -> BoundAction:
        """Reboots a server gracefully by sending an ACPI request.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.reboot(self)

    def reset(self) -> BoundAction:
        """Cuts power to a server and starts it again.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.reset(self)

    def shutdown(self) -> BoundAction:
        """Shuts down a server gracefully by sending an ACPI shutdown request.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.shutdown(self)

    def reset_password(self) -> ResetPasswordResponse:
        """Resets the root password. Only works for Linux systems that are running the qemu guest agent.

        :return: :class:`ResetPasswordResponse <hcloud.servers.domain.ResetPasswordResponse>`
        """
        return self._client.reset_password(self)

    def enable_rescue(self, type: str | None=None, ssh_keys: list[str] | None=None) -> EnableRescueResponse:
        """Enable the Hetzner Rescue System for this server.

        :param type: str
                Type of rescue system to boot (default: linux64)
                Choices: linux64, linux32, freebsd64
        :param ssh_keys: List[str]
                Array of SSH key IDs which should be injected into the rescue system. Only available for types: linux64 and linux32.
        :return: :class:`EnableRescueResponse <hcloud.servers.domain.EnableRescueResponse>`
        """
        return self._client.enable_rescue(self, type=type, ssh_keys=ssh_keys)

    def disable_rescue(self) -> BoundAction:
        """Disables the Hetzner Rescue System for a server.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.disable_rescue(self)

    def create_image(self, description: str | None=None, type: str | None=None, labels: dict[str, str] | None=None) -> CreateImageResponse:
        """Creates an image (snapshot) from a server by copying the contents of its disks.

        :param description: str (optional)
               Description of the image. If you do not set this we auto-generate one for you.
        :param type: str (optional)
               Type of image to create (default: snapshot)
               Choices: snapshot, backup
        :param labels: Dict[str, str]
               User-defined labels (key-value pairs)
        :return:  :class:`CreateImageResponse <hcloud.images.domain.CreateImageResponse>`
        """
        return self._client.create_image(self, description, type, labels)

    def rebuild(self, image: Image | BoundImage, *, return_response: bool=False) -> RebuildResponse | BoundAction:
        """Rebuilds a server overwriting its disk with the content of an image, thereby destroying all data on the target server.

        :param image: Image to use for the rebuilt server
        :param return_response: Whether to return the full response or only the action.
        """
        return self._client.rebuild(self, image, return_response=return_response)

    def change_type(self, server_type: ServerType | BoundServerType, upgrade_disk: bool) -> BoundAction:
        """Changes the type (Cores, RAM and disk sizes) of a server.

        :param server_type: :class:`BoundServerType <hcloud.server_types.client.BoundServerType>` or :class:`ServerType <hcloud.server_types.domain.ServerType>`
               Server type the server should migrate to
        :param upgrade_disk: boolean
               If false, do not upgrade the disk. This allows downgrading the server type later.
        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_type(self, server_type, upgrade_disk)

    def enable_backup(self) -> BoundAction:
        """Enables and configures the automatic daily backup option for the server. Enabling automatic backups will increase the price of the server by 20%.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.enable_backup(self)

    def disable_backup(self) -> BoundAction:
        """Disables the automatic backup option and deletes all existing Backups for a Server.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.disable_backup(self)

    def attach_iso(self, iso: Iso | BoundIso) -> BoundAction:
        """Attaches an ISO to a server.

        :param iso: :class:`BoundIso <hcloud.isos.client.BoundIso>` or :class:`Server <hcloud.isos.domain.Iso>`
        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.attach_iso(self, iso)

    def detach_iso(self) -> BoundAction:
        """Detaches an ISO from a server.

        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.detach_iso(self)

    def change_dns_ptr(self, ip: str, dns_ptr: str | None) -> BoundAction:
        """Changes the hostname that will appear when getting the hostname belonging to the primary IPs (ipv4 and ipv6) of this server.

        :param ip: str
                   The IP address for which to set the reverse DNS entry
        :param dns_ptr:
                  Hostname to set as a reverse DNS PTR entry, will reset to original default value if `None`
        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_dns_ptr(self, ip, dns_ptr)

    def change_protection(self, delete: bool | None=None, rebuild: bool | None=None) -> BoundAction:
        """Changes the protection configuration of the server.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param delete: boolean
                     If true, prevents the server from being deleted (currently delete and rebuild attribute needs to have the same value)
        :param rebuild: boolean
                     If true, prevents the server from being rebuilt (currently delete and rebuild attribute needs to have the same value)
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete, rebuild)

    def request_console(self) -> RequestConsoleResponse:
        """Requests credentials for remote access via vnc over websocket to keyboard, monitor, and mouse for a server.

        :return: :class:`RequestConsoleResponse <hcloud.servers.domain.RequestConsoleResponse>`
        """
        return self._client.request_console(self)

    def attach_to_network(self, network: Network | BoundNetwork, ip: str | None=None, alias_ips: list[str] | None=None) -> BoundAction:
        """Attaches a server to a network

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param ip: str
                IP to request to be assigned to this server
        :param alias_ips: List[str]
                New alias IPs to set for this server.
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.attach_to_network(self, network, ip, alias_ips)

    def detach_from_network(self, network: Network | BoundNetwork) -> BoundAction:
        """Detaches a server from a network.

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.detach_from_network(self, network)

    def change_alias_ips(self, network: Network | BoundNetwork, alias_ips: list[str]) -> BoundAction:
        """Changes the alias IPs of an already attached network.

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param alias_ips: List[str]
                New alias IPs to set for this server.
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_alias_ips(self, network, alias_ips)

    def add_to_placement_group(self, placement_group: PlacementGroup | BoundPlacementGroup) -> BoundAction:
        """Adds a server to a placement group.

        :param placement_group: :class:`BoundPlacementGroup <hcloud.placement_groups.client.BoundPlacementGroup>` or :class:`Network <hcloud.placement_groups.domain.PlacementGroup>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.add_to_placement_group(self, placement_group)

    def remove_from_placement_group(self) -> BoundAction:
        """Removes a server from a placement group.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.remove_from_placement_group(self)