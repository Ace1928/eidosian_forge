from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class PrimaryPreferred(_ServerMode):
    """PrimaryPreferred read preference.

    * When directly connected to one mongod queries are allowed to standalone
      servers, to a replica set primary, or to replica set secondaries.
    * When connected to a mongos queries are sent to the primary of a shard if
      available, otherwise a shard secondary.
    * When connected to a replica set queries are sent to the primary if
      available, otherwise a secondary.

    .. note:: When a :class:`~pymongo.mongo_client.MongoClient` is first
      created reads will be routed to an available secondary until the
      primary of the replica set is discovered.

    :Parameters:
      - `tag_sets`: The :attr:`~tag_sets` to use if the primary is not
        available.
      - `max_staleness`: (integer, in seconds) The maximum estimated
        length of time a replica set secondary can fall behind the primary in
        replication before it will no longer be selected for operations.
        Default -1, meaning no maximum. If it is set, it must be at least
        90 seconds.
      - `hedge`: The :attr:`~hedge` to use if the primary is not available.

    .. versionchanged:: 3.11
       Added ``hedge`` parameter.
    """
    __slots__ = ()

    def __init__(self, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        super().__init__(_PRIMARY_PREFERRED, tag_sets, max_staleness, hedge)

    def __call__(self, selection: Selection) -> Selection:
        """Apply this read preference to Selection."""
        if selection.primary:
            return selection.primary_selection
        else:
            return secondary_with_tags_server_selector(self.tag_sets, max_staleness_selectors.select(self.max_staleness, selection))