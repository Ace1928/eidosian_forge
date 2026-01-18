from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _validate_memory_limit(self, limit):
    if self._values['memory'] == 'small':
        return '0'
    if self._values['memory'] == 'medium':
        return '200'
    if self._values['memory'] == 'large':
        return '500'
    if 0 <= int(limit) <= 8192:
        return str(limit)
    raise F5ModuleError("Valid 'memory' must be in range 0 - 8192, 'small', 'medium', or 'large'.")