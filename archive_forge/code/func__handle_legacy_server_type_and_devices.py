from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _handle_legacy_server_type_and_devices(self, devices_change, server_change):
    result = {}
    if server_change and devices_change:
        result['devices'] = self.want.devices
        if len(self.want.devices) > 1 and self.want.server_type == 'bigip':
            if self.have.raw_server_type != 'redundant-bigip':
                result['server_type'] = 'redundant-bigip'
        elif self.want.server_type == 'bigip':
            if self.have.raw_server_type != 'single-bigip':
                result['server_type'] = 'single-bigip'
        else:
            result['server_type'] = self.want.server_type
    elif devices_change:
        result['devices'] = self.want.devices
        if len(self.want.devices) > 1 and self.have.server_type == 'bigip':
            if self.have.raw_server_type != 'redundant-bigip':
                result['server_type'] = 'redundant-bigip'
        elif self.have.server_type == 'bigip':
            if self.have.raw_server_type != 'single-bigip':
                result['server_type'] = 'single-bigip'
        else:
            result['server_type'] = self.want.server_type
    elif server_change:
        if len(self.have.devices) > 1 and self.want.server_type == 'bigip':
            if self.have.raw_server_type != 'redundant-bigip':
                result['server_type'] = 'redundant-bigip'
        elif self.want.server_type == 'bigip':
            if self.have.raw_server_type != 'single-bigip':
                result['server_type'] = 'single-bigip'
        else:
            result['server_type'] = self.want.server_type
    return result