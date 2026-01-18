from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _wait_for_sync(self):
    for x in range(1, 180):
        time.sleep(3)
        status = self._get_status_from_resource()
        if status in ['Changes Pending']:
            details = self._get_details_from_resource()
            self._validate_pending_status(details)
        elif status in ['Awaiting Initial Sync', 'Not All Devices Synced']:
            pass
        elif status == 'In Sync':
            return
        elif status == 'Disconnected':
            raise F5ModuleError('One or more devices are unreachable (disconnected). Resolve any communication problems before attempting to sync.')
        else:
            raise F5ModuleError(status)