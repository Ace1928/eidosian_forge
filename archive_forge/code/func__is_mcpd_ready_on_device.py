from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
def _is_mcpd_ready_on_device(self):
    try:
        command = 'tmsh show sys mcp-state | grep running'
        params = dict(command='run', utilCmdArgs='-c "{0}"'.format(command))
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
    except Exception as ex:
        if '"message":"X-F5-Auth-Token has expired."' in str(ex):
            raise
        pass
    return False