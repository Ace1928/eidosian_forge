from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_true_names(self, item):
    if 'true_names' not in item:
        return False
    result = flatten_boolean(item['true_names'])
    if result == 'yes':
        return True
    if result == 'no':
        return False