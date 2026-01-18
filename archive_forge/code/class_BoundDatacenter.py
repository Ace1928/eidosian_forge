from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from ..server_types import BoundServerType
from .domain import Datacenter, DatacenterServerTypes
class BoundDatacenter(BoundModelBase, Datacenter):
    _client: DatacentersClient
    model = Datacenter

    def __init__(self, client: DatacentersClient, data: dict):
        location = data.get('location')
        if location is not None:
            data['location'] = BoundLocation(client._client.locations, location)
        server_types = data.get('server_types')
        if server_types is not None:
            available = [BoundServerType(client._client.server_types, {'id': server_type}, complete=False) for server_type in server_types['available']]
            supported = [BoundServerType(client._client.server_types, {'id': server_type}, complete=False) for server_type in server_types['supported']]
            available_for_migration = [BoundServerType(client._client.server_types, {'id': server_type}, complete=False) for server_type in server_types['available_for_migration']]
            data['server_types'] = DatacenterServerTypes(available=available, supported=supported, available_for_migration=available_for_migration)
        super().__init__(client, data)