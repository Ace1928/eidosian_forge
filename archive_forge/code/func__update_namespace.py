import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _update_namespace(self, path, headers, data):
    response = requests.put(path, headers=headers, json=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    namespace = response.json()
    expected_namespace = {'namespace': data['namespace'], 'display_name': data['display_name'], 'description': data['description'], 'visibility': data['visibility'], 'protected': True, 'owner': data['owner'], 'self': '/v2/metadefs/namespaces/%s' % data['namespace'], 'schema': '/v2/schemas/metadefs/namespace'}
    namespace.pop('created_at')
    namespace.pop('updated_at')
    self.assertEqual(namespace, expected_namespace)
    path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    namespace = response.json()
    namespace.pop('created_at')
    namespace.pop('updated_at')
    self.assertEqual(namespace, expected_namespace)
    return namespace