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
def _wait_for_async_task_to_finish_on_device(self, task_id):
    uri = 'https://{0}:{1}/mgmt/tm/task/cli/script/{2}/result'.format(self.client.provider['server'], self.client.provider['server_port'], task_id)
    while True:
        try:
            resp = self.client.api.get(uri, timeout=10)
        except (socket.timeout, ssl.SSLError):
            continue
        try:
            response = resp.json()
        except ValueError:
            continue
        if response['_taskState'] == 'FAILED':
            raise F5ModuleError('qkview creation task failed unexpectedly.')
        if response['_taskState'] == 'COMPLETED':
            return True
        time.sleep(3)