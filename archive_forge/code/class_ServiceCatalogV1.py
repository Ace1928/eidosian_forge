import datetime
import json
import time
from urllib.parse import urljoin
from keystoneauth1 import discover
from keystoneauth1 import plugin
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.identity import base
class ServiceCatalogV1:

    def __init__(self, auth_url, storage_url, account):
        self.auth_url = auth_url
        self._storage_url = storage_url
        self._account = account

    @property
    def storage_url(self):
        if self._account:
            return urljoin(self._storage_url.rstrip('/'), self._account)
        return self._storage_url

    @property
    def catalog(self):
        endpoints = [{'region': 'default', 'publicURL': self._storage_url}]
        if self.storage_url != self._storage_url:
            endpoints.insert(0, {'region': 'override', 'publicURL': self.storage_url})
        return [{'name': 'swift', 'type': 'object-store', 'endpoints': endpoints}, {'name': 'auth', 'type': 'identity', 'endpoints': [{'region': 'default', 'publicURL': self.auth_url}]}]

    def url_for(self, **kwargs):
        return self.endpoint_data_for(**kwargs).url

    def endpoint_data_for(self, **kwargs):
        kwargs.setdefault('interface', 'public')
        kwargs.setdefault('service_type', None)
        if kwargs['service_type'] == 'object-store':
            return discover.EndpointData(service_type='object-store', service_name='swift', interface=kwargs['interface'], region_name='default', catalog_url=self.storage_url)
        if 'service_name' in kwargs and 'region_name' in kwargs:
            msg = '%(interface)s endpoint for %(service_type)s service named %(service_name)s in %(region_name)s region not found' % kwargs
        elif 'service_name' in kwargs:
            msg = '%(interface)s endpoint for %(service_type)s service named %(service_name)s not found' % kwargs
        elif 'region_name' in kwargs:
            msg = '%(interface)s endpoint for %(service_type)s service in %(region_name)s region not found' % kwargs
        else:
            msg = '%(interface)s endpoint for %(service_type)s service not found' % kwargs
        raise exceptions.EndpointNotFound(msg)