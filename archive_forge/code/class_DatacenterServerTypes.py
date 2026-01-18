from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class DatacenterServerTypes(BaseDomain):
    """DatacenterServerTypes Domain

    :param available: List[:class:`BoundServerTypes <hcloud.server_types.client.BoundServerTypes>`]
           All available server types for this datacenter
    :param supported: List[:class:`BoundServerTypes <hcloud.server_types.client.BoundServerTypes>`]
           All supported server types for this datacenter
    :param available_for_migration: List[:class:`BoundServerTypes <hcloud.server_types.client.BoundServerTypes>`]
           All available for migration (change type) server types for this datacenter
    """
    __slots__ = ('available', 'supported', 'available_for_migration')

    def __init__(self, available: list[BoundServerType], supported: list[BoundServerType], available_for_migration: list[BoundServerType]):
        self.available = available
        self.supported = supported
        self.available_for_migration = available_for_migration