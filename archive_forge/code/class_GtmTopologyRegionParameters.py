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
class GtmTopologyRegionParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'regionMembers': 'region_members'}
    returnables = ['name', 'full_path', 'region_members']

    def _string_to_dict(self, member):
        result = dict()
        item = member['name'].split(' ', 2)
        if len(item) > 2:
            result['negate'] = 'yes'
            if item[1] == 'geoip-isp':
                result['geo_isp'] = item[2]
            else:
                result[item[1]] = item[2]
            return result
        else:
            if item[0] == 'geoip-isp':
                result['geo_isp'] = item[1]
            else:
                result[item[0]] = item[1]
            return result

    @property
    def region_members(self):
        result = []
        if self._values['region_members'] is None:
            return []
        for member in self._values['region_members']:
            result.append(self._string_to_dict(member))
        return result