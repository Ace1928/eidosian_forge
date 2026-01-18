from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from warnings import warn
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Iso
class BoundIso(BoundModelBase, Iso):
    _client: IsosClient
    model = Iso