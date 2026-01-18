from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _devices_changed(self):
    if self.want.devices is None and self.want.server_type is None:
        return None
    if self.want.devices is None:
        devices = self.have.devices
    else:
        devices = self.want.devices
    if self.have.devices is None:
        have_devices = []
    else:
        have_devices = self.have.devices
    if len(devices) == 0:
        raise F5ModuleError('A GTM server must have at least one device associated with it.')
    want = [OrderedDict(sorted(d.items())) for d in devices]
    have = [OrderedDict(sorted(d.items())) for d in have_devices]
    if len(have_devices) > 0:
        if self._false_positive(devices, have_devices):
            return False
    if want != have:
        return True
    return False