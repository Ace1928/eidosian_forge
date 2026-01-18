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
class TrafficGroupsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'autoFailbackEnabled': 'auto_failback_enabled', 'autoFailbackTime': 'auto_failback_time', 'haLoadFactor': 'ha_load_factor', 'haOrder': 'ha_order', 'isFloating': 'is_floating', 'mac': 'mac_masquerade_address'}
    returnables = ['full_path', 'name', 'description', 'auto_failback_enabled', 'auto_failback_time', 'ha_load_factor', 'ha_order', 'is_floating', 'mac_masquerade_address']

    @property
    def auto_failback_time(self):
        if self._values['auto_failback_time'] is None:
            return None
        return int(self._values['auto_failback_time'])

    @property
    def auto_failback_enabled(self):
        if self._values['auto_failback_enabled'] is None:
            return None
        elif self._values['auto_failback_enabled'] == 'false':
            return 'no'
        return 'yes'

    @property
    def is_floating(self):
        if self._values['is_floating'] is None:
            return None
        elif self._values['is_floating'] == 'true':
            return 'yes'
        return 'no'

    @property
    def mac_masquerade_address(self):
        if self._values['mac_masquerade_address'] in [None, 'none']:
            return None
        return self._values['mac_masquerade_address']