import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _check_tag_access(tags, tenant):
    headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
    for tag in tags:
        for namespace, tag_name in tag.items():
            path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
            response = requests.get(path, headers=headers)
            if namespace.split('_')[1] == 'public':
                expected = http.OK
            else:
                expected = http.NOT_FOUND
            self.assertEqual(expected, response.status_code)
            path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace)
            response = requests.get(path, headers=headers)
            self.assertEqual(expected, response.status_code)
            if expected == http.OK:
                resp_props = response.json()['tags']
                self.assertEqual(sorted(tag.values()), sorted([x['name'] for x in resp_props]))