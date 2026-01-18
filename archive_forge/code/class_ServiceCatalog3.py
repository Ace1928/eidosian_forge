from troveclient.compat import exceptions
class ServiceCatalog3(object):
    """Represents a Keystone Service Catalog which describes a service.

    This class has methods to obtain a valid token as well as a public service
    url and a management url.

    """

    def __init__(self, resource_dict, region=None, service_type=None, service_name=None, service_url=None, token=None):
        self.body = resource_dict
        self.region = region
        self.service_type = service_type
        self.service_name = service_name
        self.service_url = service_url
        self.management_url = None
        self.public_url = None
        self.token = token
        self._load()

    def _load(self):
        if not self.service_url:
            self.public_url = self._url_for(attr='region', filter_value=self.region, endpoint_type='public')
            self.management_url = self._url_for(attr='region', filter_value=self.region, endpoint_type='admin')
        else:
            self.public_url = self.service_url
            self.management_url = self.service_url

    def get_token(self):
        return self.token

    def get_management_url(self):
        return self.management_url

    def get_public_url(self):
        return self.public_url

    def _url_for(self, attr=None, filter_value=None, endpoint_type='public'):
        """Fetch requested URL.

        Fetch the public URL from the Trove service for a particular
        endpoint attribute. If none given, return the first.
        """
        'Fetch the requested end point URL.\n         '
        matching_endpoints = []
        catalog = self.body['token']['catalog']
        for service in catalog:
            if service.get('type') != self.service_type:
                continue
            if self.service_name and self.service_type == 'database' and (service.get('name') != self.service_name):
                continue
            endpoints = service['endpoints']
            for endpoint in endpoints:
                if endpoint.get('interface') == endpoint_type and (not filter_value or endpoint.get(attr) == filter_value):
                    matching_endpoints.append(endpoint)
        if not matching_endpoints:
            raise exceptions.EndpointNotFound()
        else:
            return matching_endpoints[0].get('url')