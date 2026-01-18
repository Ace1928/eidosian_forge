from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _signature_set_exists_on_device(self, name):
    uri = 'https://{0}:{1}/mgmt/tm/asm/signature-sets'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = '?$select=name'
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if any((p['name'] == name for p in response['items'])):
        return True
    return False