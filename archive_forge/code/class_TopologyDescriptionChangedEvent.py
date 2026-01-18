from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class TopologyDescriptionChangedEvent(TopologyEvent):
    """Published when the topology description changes.

    .. versionadded:: 3.3
    """
    __slots__ = ('__previous_description', '__new_description')

    def __init__(self, previous_description: TopologyDescription, new_description: TopologyDescription, *args: Any) -> None:
        super().__init__(*args)
        self.__previous_description = previous_description
        self.__new_description = new_description

    @property
    def previous_description(self) -> TopologyDescription:
        """The previous
        :class:`~pymongo.topology_description.TopologyDescription`.
        """
        return self.__previous_description

    @property
    def new_description(self) -> TopologyDescription:
        """The new
        :class:`~pymongo.topology_description.TopologyDescription`.
        """
        return self.__new_description

    def __repr__(self) -> str:
        return '<{} topology_id: {} changed from: {}, to: {}>'.format(self.__class__.__name__, self.topology_id, self.previous_description, self.new_description)