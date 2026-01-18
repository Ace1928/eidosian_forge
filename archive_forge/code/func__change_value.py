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
def _change_value(self, key, value):
    if key in ['region', 'pool', 'datacenter']:
        return (key, fq_name(self.partition, value))
    if key == 'isp':
        return (key, fq_name('Common', value))
    if key == 'continent':
        return (key, self.continents.get(value, value))
    if key == 'country':
        return (key, self.countries.get(value, value))
    if key == 'geo_isp':
        return ('geoip-isp', value)
    if key == 'subnet':
        return (key, self._test_subnet(value))
    return (key, value)