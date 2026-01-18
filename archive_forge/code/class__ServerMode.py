from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class _ServerMode:
    """Base class for all read preferences."""
    __slots__ = ('__mongos_mode', '__mode', '__tag_sets', '__max_staleness', '__hedge')

    def __init__(self, mode: int, tag_sets: Optional[_TagSets]=None, max_staleness: int=-1, hedge: Optional[_Hedge]=None) -> None:
        self.__mongos_mode = _MONGOS_MODES[mode]
        self.__mode = mode
        self.__tag_sets = _validate_tag_sets(tag_sets)
        self.__max_staleness = _validate_max_staleness(max_staleness)
        self.__hedge = _validate_hedge(hedge)

    @property
    def name(self) -> str:
        """The name of this read preference."""
        return self.__class__.__name__

    @property
    def mongos_mode(self) -> str:
        """The mongos mode of this read preference."""
        return self.__mongos_mode

    @property
    def document(self) -> dict[str, Any]:
        """Read preference as a document."""
        doc: dict[str, Any] = {'mode': self.__mongos_mode}
        if self.__tag_sets not in (None, [{}]):
            doc['tags'] = self.__tag_sets
        if self.__max_staleness != -1:
            doc['maxStalenessSeconds'] = self.__max_staleness
        if self.__hedge not in (None, {}):
            doc['hedge'] = self.__hedge
        return doc

    @property
    def mode(self) -> int:
        """The mode of this read preference instance."""
        return self.__mode

    @property
    def tag_sets(self) -> _TagSets:
        """Set ``tag_sets`` to a list of dictionaries like [{'dc': 'ny'}] to
        read only from members whose ``dc`` tag has the value ``"ny"``.
        To specify a priority-order for tag sets, provide a list of
        tag sets: ``[{'dc': 'ny'}, {'dc': 'la'}, {}]``. A final, empty tag
        set, ``{}``, means "read from any member that matches the mode,
        ignoring tags." MongoClient tries each set of tags in turn
        until it finds a set of tags with at least one matching member.
        For example, to only send a query to an analytic node::

           Nearest(tag_sets=[{"node":"analytics"}])

        Or using :class:`SecondaryPreferred`::

           SecondaryPreferred(tag_sets=[{"node":"analytics"}])

           .. seealso:: `Data-Center Awareness
               <https://www.mongodb.com/docs/manual/data-center-awareness/>`_
        """
        return list(self.__tag_sets) if self.__tag_sets else [{}]

    @property
    def max_staleness(self) -> int:
        """The maximum estimated length of time (in seconds) a replica set
        secondary can fall behind the primary in replication before it will
        no longer be selected for operations, or -1 for no maximum.
        """
        return self.__max_staleness

    @property
    def hedge(self) -> Optional[_Hedge]:
        """The read preference ``hedge`` parameter.

        A dictionary that configures how the server will perform hedged reads.
        It consists of the following keys:

        - ``enabled``: Enables or disables hedged reads in sharded clusters.

        Hedged reads are automatically enabled in MongoDB 4.4+ when using a
        ``nearest`` read preference. To explicitly enable hedged reads, set
        the ``enabled`` key  to ``true``::

            >>> Nearest(hedge={'enabled': True})

        To explicitly disable hedged reads, set the ``enabled`` key  to
        ``False``::

            >>> Nearest(hedge={'enabled': False})

        .. versionadded:: 3.11
        """
        return self.__hedge

    @property
    def min_wire_version(self) -> int:
        """The wire protocol version the server must support.

        Some read preferences impose version requirements on all servers (e.g.
        maxStalenessSeconds requires MongoDB 3.4 / maxWireVersion 5).

        All servers' maxWireVersion must be at least this read preference's
        `min_wire_version`, or the driver raises
        :exc:`~pymongo.errors.ConfigurationError`.
        """
        return 0 if self.__max_staleness == -1 else 5

    def __repr__(self) -> str:
        return '{}(tag_sets={!r}, max_staleness={!r}, hedge={!r})'.format(self.name, self.__tag_sets, self.__max_staleness, self.__hedge)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _ServerMode):
            return self.mode == other.mode and self.tag_sets == other.tag_sets and (self.max_staleness == other.max_staleness) and (self.hedge == other.hedge)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __getstate__(self) -> dict[str, Any]:
        """Return value of object for pickling.

        Needed explicitly because __slots__() defined.
        """
        return {'mode': self.__mode, 'tag_sets': self.__tag_sets, 'max_staleness': self.__max_staleness, 'hedge': self.__hedge}

    def __setstate__(self, value: Mapping[str, Any]) -> None:
        """Restore from pickling."""
        self.__mode = value['mode']
        self.__mongos_mode = _MONGOS_MODES[self.__mode]
        self.__tag_sets = _validate_tag_sets(value['tag_sets'])
        self.__max_staleness = _validate_max_staleness(value['max_staleness'])
        self.__hedge = _validate_hedge(value['hedge'])

    def __call__(self, selection: Selection) -> Selection:
        return selection