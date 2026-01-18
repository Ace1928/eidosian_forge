from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
class InsertManyResult(_WriteResult):
    """The return type for :meth:`~pymongo.collection.Collection.insert_many`."""
    __slots__ = ('__inserted_ids',)

    def __init__(self, inserted_ids: list[Any], acknowledged: bool) -> None:
        self.__inserted_ids = inserted_ids
        super().__init__(acknowledged)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__inserted_ids!r}, acknowledged={self.acknowledged})'

    @property
    def inserted_ids(self) -> list[Any]:
        """A list of _ids of the inserted documents, in the order provided.

        .. note:: If ``False`` is passed for the `ordered` parameter to
          :meth:`~pymongo.collection.Collection.insert_many` the server
          may have inserted the documents in a different order than what
          is presented here.
        """
        return self.__inserted_ids