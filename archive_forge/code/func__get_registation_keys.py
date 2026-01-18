from __future__ import (absolute_import, division, print_function)
import random
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible_collections.f5networks.f5_modules.plugins.module_utils.bigiq import F5RestClient
def _get_registation_keys(self, pool_id):
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/regkey/licenses/{2}/offerings/'.format(self.host, self.port, pool_id)
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
    regkeys = [x['regKey'] for x in response['items']]
    if not regkeys:
        raise AnsibleError('Failed to obtain registration keys')
    return regkeys