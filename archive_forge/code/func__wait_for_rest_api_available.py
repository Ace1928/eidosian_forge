from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _wait_for_rest_api_available(self):
    nops = 0
    time.sleep(5)
    while nops < 3:
        try:
            if self._is_rest_available():
                nops += 1
            else:
                nops = 0
                time.sleep(5)
        except Exception:
            try:
                self.client.reconnect()
            except Exception:
                pass