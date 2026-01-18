from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from ..server_types import BoundServerType
from .domain import Datacenter, DatacenterServerTypes
class DatacentersPageResult(NamedTuple):
    datacenters: list[BoundDatacenter]
    meta: Meta | None