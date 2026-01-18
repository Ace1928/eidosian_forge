from __future__ import (absolute_import, division, print_function)
import random
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible_collections.f5networks.f5_modules.plugins.module_utils.bigiq import F5RestClient
def _get_pool_uuid(self):
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/regkey/licenses'.format(self.host, self.port)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise AnsibleError(str(ex))
    if 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise AnsibleError(response['message'])
        else:
            raise AnsibleError(resp.content)
    if 'items' not in response:
        raise AnsibleError('No license pools configured on BIGIQ')
    resource = next((x for x in response['items'] if x['name'] == self.pool_name), None)
    if resource is None:
        raise AnsibleError('Could not find the specified license pool.')
    return resource['id']