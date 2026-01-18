from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..locations import BoundLocation
from ..server_types import BoundServerType
from .domain import Datacenter, DatacenterServerTypes
class DatacentersClient(ClientEntityBase):
    _client: Client

    def get_by_id(self, id: int) -> BoundDatacenter:
        """Get a specific datacenter by its ID.

        :param id: int
        :return: :class:`BoundDatacenter <hcloud.datacenters.client.BoundDatacenter>`
        """
        response = self._client.request(url=f'/datacenters/{id}', method='GET')
        return BoundDatacenter(self, response['datacenter'])

    def get_list(self, name: str | None=None, page: int | None=None, per_page: int | None=None) -> DatacentersPageResult:
        """Get a list of datacenters

        :param name: str (optional)
               Can be used to filter datacenters by their name.
        :param page: int (optional)
               Specifies the page to fetch
        :param per_page: int (optional)
               Specifies how many results are returned by page
        :return: (List[:class:`BoundDatacenter <hcloud.datacenters.client.BoundDatacenter>`], :class:`Meta <hcloud.core.domain.Meta>`)
        """
        params: dict[str, Any] = {}
        if name is not None:
            params['name'] = name
        if page is not None:
            params['page'] = page
        if per_page is not None:
            params['per_page'] = per_page
        response = self._client.request(url='/datacenters', method='GET', params=params)
        datacenters = [BoundDatacenter(self, datacenter_data) for datacenter_data in response['datacenters']]
        return DatacentersPageResult(datacenters, Meta.parse_meta(response))

    def get_all(self, name: str | None=None) -> list[BoundDatacenter]:
        """Get all datacenters

        :param name: str (optional)
               Can be used to filter datacenters by their name.
        :return: List[:class:`BoundDatacenter <hcloud.datacenters.client.BoundDatacenter>`]
        """
        return self._iter_pages(self.get_list, name=name)

    def get_by_name(self, name: str) -> BoundDatacenter | None:
        """Get datacenter by name

        :param name: str
               Used to get datacenter by name.
        :return: :class:`BoundDatacenter <hcloud.datacenters.client.BoundDatacenter>`
        """
        return self._get_first_by(name=name)