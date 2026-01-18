from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import ServerType
class BoundServerType(BoundModelBase, ServerType):
    _client: ServerTypesClient
    model = ServerType