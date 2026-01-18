from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class TopologyListener(_EventListener):
    """Abstract base class for topology monitoring listeners.
    Handles `TopologyOpenedEvent`, `TopologyDescriptionChangedEvent`, and
    `TopologyClosedEvent`.

    .. versionadded:: 3.3
    """

    def opened(self, event: TopologyOpenedEvent) -> None:
        """Abstract method to handle a `TopologyOpenedEvent`.

        :Parameters:
          - `event`: An instance of :class:`TopologyOpenedEvent`.
        """
        raise NotImplementedError

    def description_changed(self, event: TopologyDescriptionChangedEvent) -> None:
        """Abstract method to handle a `TopologyDescriptionChangedEvent`.

        :Parameters:
          - `event`: An instance of :class:`TopologyDescriptionChangedEvent`.
        """
        raise NotImplementedError

    def closed(self, event: TopologyClosedEvent) -> None:
        """Abstract method to handle a `TopologyClosedEvent`.

        :Parameters:
          - `event`: An instance of :class:`TopologyClosedEvent`.
        """
        raise NotImplementedError