from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _change_rate_limit_value(self, value):
    if value == 4294967295:
        return 'indefinite'
    else:
        return value