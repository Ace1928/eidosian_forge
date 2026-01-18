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
class TrunksParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'media': 'media_speed', 'lacpMode': 'lacp_mode', 'lacp': 'lacp_state', 'lacpTimeout': 'lacp_timeout', 'stp': 'stp_enabled', 'workingMbrCount': 'operational_member_count', 'linkSelectPolicy': 'link_selection_policy', 'distributionHash': 'distribution_hash', 'cfgMbrCount': 'configured_member_count'}
    returnables = ['full_path', 'name', 'description', 'media_speed', 'lacp_mode', 'lacp_enabled', 'stp_enabled', 'operational_member_count', 'media_status', 'link_selection_policy', 'lacp_timeout', 'interfaces', 'distribution_hash', 'configured_member_count']

    @property
    def lacp_enabled(self):
        if self._values['lacp_enabled'] is None:
            return None
        elif self._values['lacp_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def stp_enabled(self):
        if self._values['stp_enabled'] is None:
            return None
        elif self._values['stp_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def media_status(self):
        return self._values['stats']['status']