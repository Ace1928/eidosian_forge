import hmac
import time
import hashlib
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Container, StorageDriver
class NimbusStorageDriver(StorageDriver):
    name = 'Nimbus.io'
    website = 'https://nimbus.io/'
    connectionCls = NimbusConnection

    def __init__(self, *args, **kwargs):
        self.user_id = kwargs['user_id']
        super().__init__(*args, **kwargs)

    def iterate_containers(self):
        response = self.connection.request('/customers/%s/collections' % self.user_id)
        return self._to_containers(response.object)

    def create_container(self, container_name):
        params = {'action': 'create', 'name': container_name}
        response = self.connection.request('/customers/%s/collections' % self.user_id, params=params, method='POST')
        return self._to_container(response.object)

    def _to_containers(self, data):
        for item in data:
            yield self._to_container(item)

    def _to_container(self, data):
        name = data[0]
        extra = {'date_created': data[2]}
        return Container(name=name, extra=extra, driver=self)

    def _ex_connection_class_kwargs(self):
        result = {'id': self.user_id}
        return result