from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class NetworkTimeout(AutoReconnect):
    """An operation on an open connection exceeded socketTimeoutMS.

    The remaining connections in the pool stay open. In the case of a write
    operation, you cannot know whether it succeeded or failed.

    Subclass of :exc:`~pymongo.errors.AutoReconnect`.
    """

    @property
    def timeout(self) -> bool:
        return True