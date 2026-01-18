from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _wait_for_asm_ready(self):
    """Waits specifically for ASM

        On older versions, ASM can take longer to actually start up than
        all the previous checks take. This check here is specifically waiting for
        the Policies API to stop raising errors
        :return:
        """
    nops = 0
    restarted_asm = False
    while nops < 3:
        try:
            uri = 'https://{0}:{1}/mgmt/tm/asm/policies/'.format(self.client.provider['server'], self.client.provider['server_port'])
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in response and response['code'] in [400, 403]:
                if 'message' in response:
                    raise F5ModuleError(response['message'])
                else:
                    raise F5ModuleError(resp.content)
            if len(response['items']) >= 0:
                nops += 1
            else:
                nops = 0
        except Exception:
            if not restarted_asm:
                self._restart_asm()
                restarted_asm = True
        time.sleep(5)