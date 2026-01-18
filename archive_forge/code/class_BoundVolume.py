from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from .domain import CreateVolumeResponse, Volume
class BoundVolume(BoundModelBase, Volume):
    _client: VolumesClient
    model = Volume

    def __init__(self, client: VolumesClient, data: dict, complete: bool=True):
        location = data.get('location')
        if location is not None:
            data['location'] = BoundLocation(client._client.locations, location)
        from ..servers import BoundServer
        server = data.get('server')
        if server is not None:
            data['server'] = BoundServer(client._client.servers, {'id': server}, complete=False)
        super().__init__(client, data, complete)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a volume.

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
        """Returns all action objects for a volume.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundVolume:
        """Updates the volume properties.

        :param name: str (optional)
               New volume name
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.update(self, name, labels)

    def delete(self) -> bool:
        """Deletes a volume. All volume data is irreversibly destroyed. The volume must not be attached to a server and it must not have delete protection enabled.

        :return: boolean
        """
        return self._client.delete(self)

    def attach(self, server: Server | BoundServer, automount: bool | None=None) -> BoundAction:
        """Attaches a volume to a server. Works only if the server is in the same location as the volume.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :param automount: boolean
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.attach(self, server, automount)

    def detach(self) -> BoundAction:
        """Detaches a volume from the server it’s attached to. You may attach it to a server again at a later time.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.detach(self)

    def resize(self, size: int) -> BoundAction:
        """Changes the size of a volume. Note that downsizing a volume is not possible.

        :param size: int
               New volume size in GB (must be greater than current size)
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.resize(self, size)

    def change_protection(self, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of a volume.

        :param delete: boolean
               If True, prevents the volume from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete)