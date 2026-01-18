from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _create_temporary_cli_script_on_device(self, args):
    uri = 'https://{0}:{1}/mgmt/tm/cli/script'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
        if 'code' in response and response['code'] in [404, 409]:
            return False
    except ValueError:
        pass
    if resp.status in [404, 409]:
        return False
    return True