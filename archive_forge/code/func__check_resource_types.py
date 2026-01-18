import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _check_resource_types(tenant, total_rs_types):
    path = self._url('/v2/metadefs/resource_types')
    headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    metadef_resource_type = response.json()
    self.assertEqual(sorted((x['name'] for x in metadef_resource_type['resource_types'])), sorted((value for x in total_rs_types for key, value in x.items())))