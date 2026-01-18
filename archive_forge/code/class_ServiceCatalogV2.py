import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
class ServiceCatalogV2(ServiceCatalog):
    """An object for encapsulating the v2 service catalog.

    The object is created using raw v2 auth token from Keystone.
    """

    @classmethod
    def from_token(cls, token):
        if 'access' not in token:
            raise ValueError('Invalid token format for fetching catalog')
        return cls(token['access'].get('serviceCatalog', {}))

    @staticmethod
    def normalize_interface(interface):
        if interface and 'URL' not in interface:
            interface += 'URL'
        return interface

    def is_interface_match(self, endpoint, interface):
        return interface in endpoint

    def _normalize_endpoints(self, endpoints):
        """Translate endpoint description dicts into v3 form.

        Takes a raw endpoint description from the catalog and changes
        it to be in v3 format. It also saves a copy of the data in
        raw_endpoint so that it can be returned by methods that expect the
        actual original data.

        :param list endpoints: List of endpoint description dicts

        :returns: List of endpoint description dicts in v3 format
        """
        new_endpoints = []
        for endpoint in endpoints:
            raw_endpoint = endpoint.copy()
            interface_urls = {}
            interface_keys = [key for key in endpoint.keys() if key.endswith('URL')]
            for key in interface_keys:
                interface = self.normalize_interface(key)
                interface_urls[interface] = endpoint.pop(key)
            for interface, url in interface_urls.items():
                new_endpoint = endpoint.copy()
                new_endpoint['interface'] = interface
                new_endpoint['url'] = url
                new_endpoint['raw_endpoint'] = raw_endpoint
                new_endpoints.append(new_endpoint)
        return new_endpoints

    def _denormalize_endpoints(self, endpoints):
        """Return original endpoint description dicts.

        Takes a list of EndpointData objects and returns the original
        dict that was returned from the catalog.

        :param list endpoints: List of `keystoneauth1.discover.EndpointData`

        :returns: List of endpoint description dicts in original catalog format
        """
        raw_endpoints = super(ServiceCatalogV2, self)._denormalize_endpoints(endpoints)
        seen = {}
        endpoints = []
        for endpoint in raw_endpoints:
            if str(endpoint) in seen:
                continue
            seen[str(endpoint)] = True
            endpoints.append(endpoint)
        return endpoints