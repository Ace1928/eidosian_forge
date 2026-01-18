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
class DevicesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'activeModules': 'active_modules', 'baseMac': 'base_mac_address', 'chassisId': 'chassis_id', 'chassisType': 'chassis_type', 'configsyncIp': 'configsync_address', 'failoverState': 'failover_state', 'managementIp': 'management_address', 'marketingName': 'marketing_name', 'multicastIp': 'multicast_address', 'optionalModules': 'optional_modules', 'platformId': 'platform_id', 'mirrorIp': 'primary_mirror_address', 'mirrorSecondaryIp': 'secondary_mirror_address', 'version': 'software_version', 'timeLimitedModules': 'timelimited_modules', 'timeZone': 'timezone', 'unicastAddress': 'unicast_addresses', 'selfDevice': 'self'}
    returnables = ['full_path', 'name', 'active_modules', 'base_mac_address', 'build', 'chassis_id', 'chassis_type', 'comment', 'configsync_address', 'contact', 'description', 'edition', 'failover_state', 'hostname', 'location', 'management_address', 'marketing_name', 'multicast_address', 'optional_modules', 'platform_id', 'primary_mirror_address', 'product', 'secondary_mirror_address', 'self', 'software_version', 'timelimited_modules', 'timezone', 'unicast_addresses']

    @property
    def active_modules(self):
        if self._values['active_modules'] is None:
            return None
        result = []
        for x in self._values['active_modules']:
            parts = x.split('|')
            result += parts[2:]
        return list(set(result))

    @property
    def self(self):
        result = flatten_boolean(self._values['self'])
        return result

    @property
    def configsync_address(self):
        if self._values['configsync_address'] in [None, 'none']:
            return None
        return self._values['configsync_address']

    @property
    def primary_mirror_address(self):
        if self._values['primary_mirror_address'] in [None, 'any6']:
            return None
        return self._values['primary_mirror_address']

    @property
    def secondary_mirror_address(self):
        if self._values['secondary_mirror_address'] in [None, 'any6']:
            return None
        return self._values['secondary_mirror_address']

    @property
    def unicast_addresses(self):
        if self._values['unicast_addresses'] is None:
            return None
        result = []
        for addr in self._values['unicast_addresses']:
            tmp = {}
            for key in ['effectiveIp', 'effectivePort', 'ip', 'port']:
                if key in addr:
                    renamed_key = self.convert(key)
                    tmp[renamed_key] = addr.get(key, None)
            if tmp:
                result.append(tmp)
        if result:
            return result

    def convert(self, name):
        s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
        return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()