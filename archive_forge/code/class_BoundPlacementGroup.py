from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePlacementGroupResponse, PlacementGroup
class BoundPlacementGroup(BoundModelBase, PlacementGroup):
    _client: PlacementGroupsClient
    model = PlacementGroup

    def update(self, labels: dict[str, str] | None=None, name: str | None=None) -> BoundPlacementGroup:
        """Updates the name or labels of a Placement Group

        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str, (optional)
               New Name to set
        :return: :class:`BoundPlacementGroup <hcloud.placement_groups.client.BoundPlacementGroup>`
        """
        return self._client.update(self, labels, name)

    def delete(self) -> bool:
        """Deletes a Placement Group

        :return: boolean
        """
        return self._client.delete(self)