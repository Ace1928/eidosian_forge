from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def _handle_actions(self, actions):
    result = []
    if actions is None or 'items' not in actions:
        return result
    exclude_keys = ['poolReference', 'name']
    for action in actions['items']:
        tmp = dict(((k, v) for k, v in iteritems(action) if v != 0 and k not in exclude_keys))
        self._remove_internal_keywords(tmp)
        result.append(self._filter_params(tmp))
    return result