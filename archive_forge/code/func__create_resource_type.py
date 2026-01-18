import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _create_resource_type(self, namespaces):
    resource_types = []
    for namespace in namespaces:
        headers = self._headers({'X-Tenant-Id': namespace['owner']})
        data = {'name': 'resource_type_of_%s' % namespace['namespace'], 'prefix': 'hw_', 'properties_target': 'image'}
        path = self._url('/v2/metadefs/namespaces/%s/resource_types' % namespace['namespace'])
        response = requests.post(path, headers=headers, json=data)
        self.assertEqual(http.CREATED, response.status_code)
        rs_type = response.json()
        resource_type = dict()
        resource_type[namespace['namespace']] = rs_type['name']
        resource_types.append(resource_type)
    return resource_types