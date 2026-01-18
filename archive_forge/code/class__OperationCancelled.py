from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class _OperationCancelled(AutoReconnect):
    """Internal error raised when a socket operation is cancelled."""