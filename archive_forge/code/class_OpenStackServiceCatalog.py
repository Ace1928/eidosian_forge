import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackServiceCatalog:
    """
    http://docs.openstack.org/api/openstack-identity-service/2.0/content/

    This class should be instantiated with the contents of the
    'serviceCatalog' in the auth response. This will do the work of figuring
    out which services actually exist in the catalog as well as split them up
    by type, name, and region if available
    """
    _auth_version = None
    _service_catalog = None

    def __init__(self, service_catalog, auth_version=AUTH_API_VERSION):
        self._auth_version = auth_version
        if '3.x' in self._auth_version:
            entries = self._parse_service_catalog_auth_v3(service_catalog=service_catalog)
        elif '2.0' in self._auth_version:
            entries = self._parse_service_catalog_auth_v2(service_catalog=service_catalog)
        elif '1.1' in self._auth_version or '1.0' in self._auth_version:
            entries = self._parse_service_catalog_auth_v1(service_catalog=service_catalog)
        else:
            raise LibcloudError('auth version "%s" not supported' % self._auth_version)
        entries = sorted(entries, key=lambda x: x.service_type + (x.service_name or ''))
        self._entries = entries

    def get_entries(self):
        """
        Return all the entries for this service catalog.

        :rtype: ``list`` of :class:`.OpenStackServiceCatalogEntry`
        """
        return self._entries

    def get_catalog(self):
        """
        Deprecated in the favor of ``get_entries`` method.
        """
        return self.get_entries()

    def get_public_urls(self, service_type=None, name=None):
        """
        Retrieve all the available public (external) URLs for the provided
        service type and name.
        """
        endpoints = self.get_endpoints(service_type=service_type, name=name)
        result = []
        for endpoint in endpoints:
            endpoint_type = endpoint.endpoint_type
            if endpoint_type == OpenStackIdentityEndpointType.EXTERNAL:
                result.append(endpoint.url)
        return result

    def get_endpoints(self, service_type=None, name=None):
        """
        Retrieve all the endpoints for the provided service type and name.

        :rtype: ``list`` of :class:`.OpenStackServiceCatalogEntryEndpoint`
        """
        endpoints = []
        for entry in self._entries:
            if service_type and entry.service_type != service_type:
                continue
            if name and entry.service_name != name:
                continue
            for endpoint in entry.endpoints:
                endpoints.append(endpoint)
        return endpoints

    def get_endpoint(self, service_type=None, name=None, region=None, endpoint_type=OpenStackIdentityEndpointType.EXTERNAL):
        """
        Retrieve a single endpoint using the provided criteria.

        Note: If no or more than one matching endpoint is found, an exception
        is thrown.
        """
        endpoints = []
        for entry in self._entries:
            if service_type and entry.service_type != service_type:
                continue
            if name and entry.service_name != name:
                continue
            for endpoint in entry.endpoints:
                if region and endpoint.region != region:
                    continue
                if endpoint_type and endpoint.endpoint_type != endpoint_type:
                    continue
                endpoints.append(endpoint)
        if len(endpoints) == 1:
            return endpoints[0]
        elif len(endpoints) > 1:
            raise ValueError('Found more than 1 matching endpoint')
        else:
            raise LibcloudError('Could not find specified endpoint')

    def get_regions(self, service_type=None):
        """
        Retrieve a list of all the available regions.

        :param service_type: If specified, only return regions for this
                             service type.
        :type service_type: ``str``

        :rtype: ``list`` of ``str``
        """
        regions = set()
        for entry in self._entries:
            if service_type and entry.service_type != service_type:
                continue
            for endpoint in entry.endpoints:
                if endpoint.region:
                    regions.add(endpoint.region)
        return sorted(list(regions))

    def get_service_types(self, region=None):
        """
        Retrieve all the available service types.

        :param region: Optional region to retrieve service types for.
        :type region: ``str``

        :rtype: ``list`` of ``str``
        """
        service_types = set()
        for entry in self._entries:
            include = True
            for endpoint in entry.endpoints:
                if region and endpoint.region != region:
                    include = False
                    break
            if include:
                service_types.add(entry.service_type)
        return sorted(list(service_types))

    def get_service_names(self, service_type=None, region=None):
        """
        Retrieve list of service names that match service type and region.

        :type service_type: ``str``
        :type region: ``str``

        :rtype: ``list`` of ``str``
        """
        names = set()
        if '2.0' not in self._auth_version:
            raise ValueError('Unsupported version: %s' % self._auth_version)
        for entry in self._entries:
            if service_type and entry.service_type != service_type:
                continue
            include = True
            for endpoint in entry.endpoints:
                if region and endpoint.region != region:
                    include = False
                    break
            if include and entry.service_name:
                names.add(entry.service_name)
        return sorted(list(names))

    def _parse_service_catalog_auth_v1(self, service_catalog):
        entries = []
        for service, endpoints in service_catalog.items():
            entry_endpoints = []
            for endpoint in endpoints:
                region = endpoint.get('region', None)
                public_url = endpoint.get('publicURL', None)
                private_url = endpoint.get('internalURL', None)
                if public_url:
                    entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=public_url, endpoint_type=OpenStackIdentityEndpointType.EXTERNAL)
                    entry_endpoints.append(entry_endpoint)
                if private_url:
                    entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=private_url, endpoint_type=OpenStackIdentityEndpointType.INTERNAL)
                    entry_endpoints.append(entry_endpoint)
            entry = OpenStackServiceCatalogEntry(service_type=service, endpoints=entry_endpoints)
            entries.append(entry)
        return entries

    def _parse_service_catalog_auth_v2(self, service_catalog):
        entries = []
        for service in service_catalog:
            service_type = service['type']
            service_name = service.get('name', None)
            entry_endpoints = []
            for endpoint in service.get('endpoints', []):
                region = endpoint.get('region', None)
                public_url = endpoint.get('publicURL', None)
                private_url = endpoint.get('internalURL', None)
                if public_url:
                    entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=public_url, endpoint_type=OpenStackIdentityEndpointType.EXTERNAL)
                    entry_endpoints.append(entry_endpoint)
                if private_url:
                    entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=private_url, endpoint_type=OpenStackIdentityEndpointType.INTERNAL)
                    entry_endpoints.append(entry_endpoint)
            entry = OpenStackServiceCatalogEntry(service_type=service_type, endpoints=entry_endpoints, service_name=service_name)
            entries.append(entry)
        return entries

    def _parse_service_catalog_auth_v3(self, service_catalog):
        entries = []
        for item in service_catalog:
            service_type = item['type']
            service_name = item.get('name', None)
            entry_endpoints = []
            for endpoint in item['endpoints']:
                region = endpoint.get('region', None)
                url = endpoint['url']
                endpoint_type = endpoint['interface']
                if endpoint_type == 'internal':
                    endpoint_type = OpenStackIdentityEndpointType.INTERNAL
                elif endpoint_type == 'public':
                    endpoint_type = OpenStackIdentityEndpointType.EXTERNAL
                elif endpoint_type == 'admin':
                    endpoint_type = OpenStackIdentityEndpointType.ADMIN
                entry_endpoint = OpenStackServiceCatalogEntryEndpoint(region=region, url=url, endpoint_type=endpoint_type)
                entry_endpoints.append(entry_endpoint)
            entry = OpenStackServiceCatalogEntry(service_type=service_type, service_name=service_name, endpoints=entry_endpoints)
            entries.append(entry)
        return entries