from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def _wait_for_task(self, uri):
    while True:
        resp = self.client.api.get(uri)
        if resp.status == 401:
            self.client.reconnect()
            resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        if response['status'] in ['FINISHED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)
    if response['status'] == 'FAILED':
        raise F5ModuleError(response['errorMessage'])
    if response['status'] == 'CANCELLED':
        raise F5ModuleError('The task process has been cancelled.')
    if response['status'] == 'FINISHED':
        return True