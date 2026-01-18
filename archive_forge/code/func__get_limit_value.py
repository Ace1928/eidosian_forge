from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _get_limit_value(self, type):
    if self._values['limits'] is None:
        return None
    if self._values['limits'][type] is None:
        return None
    return int(self._values['limits'][type])