from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
class VlansParameters(BaseParameters):
    api_map = {'autoLasthop': 'auto_lasthop', 'cmpHash': 'cmp_hash_algorithm', 'failsafeAction': 'failsafe_action', 'failsafe': 'failsafe_enabled', 'failsafeTimeout': 'failsafe_timeout', 'ifIndex': 'if_index', 'learning': 'learning_mode', 'interfacesReference': 'interfaces', 'sourceChecking': 'source_check_enabled', 'fullPath': 'full_path'}
    returnables = ['full_path', 'name', 'auto_lasthop', 'cmp_hash_algorithm', 'description', 'failsafe_action', 'failsafe_enabled', 'failsafe_timeout', 'if_index', 'learning_mode', 'interfaces', 'mtu', 'sflow_poll_interval', 'sflow_poll_interval_global', 'sflow_sampling_rate', 'sflow_sampling_rate_global', 'source_check_enabled', 'true_mac_address', 'tag']

    @property
    def interfaces(self):
        if self._values['interfaces'] is None:
            return None
        if 'items' not in self._values['interfaces']:
            return None
        result = []
        for item in self._values['interfaces']['items']:
            tmp = dict(name=item['name'], full_path=item['fullPath'])
            if 'tagged' in item:
                tmp['tagged'] = 'yes'
            else:
                tmp['tagged'] = 'no'
            result.append(tmp)
        return result

    @property
    def sflow_poll_interval(self):
        return int(self._values['sflow']['pollInterval'])

    @property
    def sflow_poll_interval_global(self):
        return flatten_boolean(self._values['sflow']['pollIntervalGlobal'])

    @property
    def sflow_sampling_rate(self):
        return int(self._values['sflow']['samplingRate'])

    @property
    def sflow_sampling_rate_global(self):
        return flatten_boolean(self._values['sflow']['samplingRateGlobal'])

    @property
    def source_check_state(self):
        return flatten_boolean(self._values['source_check_state'])

    @property
    def true_mac_address(self):
        if self._values['stats']['macTrue'] in [None, 'none']:
            return None
        return self._values['stats']['macTrue']

    @property
    def tag(self):
        return self._values['stats']['id']

    @property
    def failsafe_enabled(self):
        return flatten_boolean(self._values['failsafe_enabled'])