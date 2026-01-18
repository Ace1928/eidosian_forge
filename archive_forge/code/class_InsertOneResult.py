from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
class InsertOneResult(_WriteResult):
    """The return type for :meth:`~pymongo.collection.Collection.insert_one`."""
    __slots__ = ('__inserted_id',)

    def __init__(self, inserted_id: Any, acknowledged: bool) -> None:
        self.__inserted_id = inserted_id
        super().__init__(acknowledged)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__inserted_id!r}, acknowledged={self.acknowledged})'

    @property
    def inserted_id(self) -> Any:
        """The inserted document's _id."""
        return self.__inserted_id