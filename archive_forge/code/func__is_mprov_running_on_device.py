from __future__ import absolute_import, division, print_function
import copy
import datetime
import signal
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.teem import send_teem
def _is_mprov_running_on_device(self):
    params = {'command': 'run', 'utilCmdArgs': '-c "ps aux | grep \'[m]prov\'"'}
    uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'commandResult' in response:
        return True
    return False