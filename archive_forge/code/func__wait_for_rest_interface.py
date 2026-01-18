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
def _wait_for_rest_interface(self):
    nops = 0
    time.sleep(5)
    while nops < 4:
        if not self._rest_endpoints_ready():
            nops += 1
        else:
            break
    time.sleep(10)