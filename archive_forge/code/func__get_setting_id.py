from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _get_setting_id(self):
    uri = 'https://{0}:{1}/mgmt/tm/asm/advanced-settings/'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = "?$filter=name+eq+'{0}'&$select=id".format(self.want.name)
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    errors = [401, 403, 409, 500, 501, 502, 503, 504]
    if resp.status in errors or ('code' in response and response['code'] in errors):
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    if 'items' in response and response['items'] != []:
        self.setting_id = response['items'][0]['id']
    if not self.setting_id:
        raise F5ModuleError('The setting: {0} was not found.'.format(self.want.name))