from __future__ import absolute_import, division, print_function
import copy
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip_network
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@staticmethod
def _flatten_negate(item):
    result = flatten_boolean(item['negate'])
    item.pop('negate')
    if result == 'yes':
        return 'not'
    return None