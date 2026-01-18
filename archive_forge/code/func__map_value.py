from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _map_value(self, item, t=False):
    if not t:
        for k in self.event_map.keys():
            if k in item:
                return k
    else:
        for k in self.uri_type_map.keys():
            if k in item:
                return k
    return None