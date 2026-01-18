from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _handle_current_server_type_and_devices(self, devices_change, server_change):
    result = {}
    if devices_change:
        result['devices'] = self.want.devices
    if server_change:
        result['server_type'] = self.want.server_type
    return result