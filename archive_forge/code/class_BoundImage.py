from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Image
class BoundImage(BoundModelBase, Image):
    _client: ImagesClient
    model = Image

    def __init__(self, client: ImagesClient, data: dict):
        from ..servers import BoundServer
        created_from = data.get('created_from')
        if created_from is not None:
            data['created_from'] = BoundServer(client._client.servers, created_from, complete=False)
        bound_to = data.get('bound_to')
        if bound_to is not None:
            data['bound_to'] = BoundServer(client._client.servers, {'id': bound_to}, complete=False)
        super().__init__(client, data)

    def get_actions_list(self, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None, status: list[str] | None=None) -> ActionsPageResult:
        """Returns a list of action objects for the image.

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
        return self._client.get_actions_list(self, sort=sort, page=page, per_page=per_page, status=status)

    def get_actions(self, sort: list[str] | None=None, status: list[str] | None=None) -> list[BoundAction]:
        """Returns all action objects for the image.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status=status, sort=sort)

    def update(self, description: str | None=None, type: str | None=None, labels: dict[str, str] | None=None) -> BoundImage:
        """Updates the Image. You may change the description, convert a Backup image to a Snapshot Image or change the image labels.

        :param description: str (optional)
               New description of Image
        :param type: str (optional)
               Destination image type to convert to
               Choices: snapshot
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundImage <hcloud.images.client.BoundImage>`
        """
        return self._client.update(self, description, type, labels)

    def delete(self) -> bool:
        """Deletes an Image. Only images of type snapshot and backup can be deleted.

        :return: bool
        """
        return self._client.delete(self)

    def change_protection(self, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of the image. Can only be used on snapshots.

        :param delete: bool
               If true, prevents the snapshot from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete)