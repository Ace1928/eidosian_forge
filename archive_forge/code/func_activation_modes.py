from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def activation_modes(self):
    value = self._values['activation_modes']
    if value is None:
        return None
    if is_empty_list(value):
        raise F5ModuleError('Activation Modes cannot be empty, please provide a value')
    return value