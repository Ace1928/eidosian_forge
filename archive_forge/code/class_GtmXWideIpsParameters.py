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
class GtmXWideIpsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'failureRcode': 'failure_rcode', 'failureRcodeResponse': 'failure_rcode_response', 'failureRcodeTtl': 'failure_rcode_ttl', 'lastResortPool': 'last_resort_pool', 'minimalResponse': 'minimal_response', 'persistCidrIpv4': 'persist_cidr_ipv4', 'persistCidrIpv6': 'persist_cidr_ipv6', 'poolLbMode': 'pool_lb_mode', 'ttlPersistence': 'ttl_persistence'}
    returnables = ['full_path', 'description', 'enabled', 'disabled', 'failure_rcode', 'failure_rcode_response', 'failure_rcode_ttl', 'last_resort_pool', 'minimal_response', 'name', 'persist_cidr_ipv4', 'persist_cidr_ipv6', 'pool_lb_mode', 'ttl_persistence', 'pools']

    @property
    def pools(self):
        result = []
        if self._values['pools'] is None:
            return []
        for pool in self._values['pools']:
            del pool['nameReference']
            for x in ['order', 'ratio']:
                if x in pool:
                    pool[x] = int(pool[x])
            result.append(pool)
        return result

    @property
    def failure_rcode_response(self):
        return flatten_boolean(self._values['failure_rcode_response'])

    @property
    def failure_rcode_ttl(self):
        if self._values['failure_rcode_ttl'] is None:
            return None
        return int(self._values['failure_rcode_ttl'])

    @property
    def persist_cidr_ipv4(self):
        if self._values['persist_cidr_ipv4'] is None:
            return None
        return int(self._values['persist_cidr_ipv4'])

    @property
    def persist_cidr_ipv6(self):
        if self._values['persist_cidr_ipv6'] is None:
            return None
        return int(self._values['persist_cidr_ipv6'])

    @property
    def ttl_persistence(self):
        if self._values['ttl_persistence'] is None:
            return None
        return int(self._values['ttl_persistence'])