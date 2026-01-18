from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ServerDescriptionChangedEvent(_ServerEvent):
    """Published when server description changes.

    .. versionadded:: 3.3
    """
    __slots__ = ('__previous_description', '__new_description')

    def __init__(self, previous_description: ServerDescription, new_description: ServerDescription, *args: Any) -> None:
        super().__init__(*args)
        self.__previous_description = previous_description
        self.__new_description = new_description

    @property
    def previous_description(self) -> ServerDescription:
        """The previous
        :class:`~pymongo.server_description.ServerDescription`.
        """
        return self.__previous_description

    @property
    def new_description(self) -> ServerDescription:
        """The new
        :class:`~pymongo.server_description.ServerDescription`.
        """
        return self.__new_description

    def __repr__(self) -> str:
        return '<{} {} changed from: {}, to: {}>'.format(self.__class__.__name__, self.server_address, self.previous_description, self.new_description)