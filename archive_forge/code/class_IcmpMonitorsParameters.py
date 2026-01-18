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
class IcmpMonitorsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'defaultsFrom': 'parent', 'adaptiveDivergenceType': 'adaptive_divergence_type', 'adaptiveDivergenceValue': 'adaptive_divergence_value', 'adaptiveLimit': 'adaptive_limit', 'adaptiveSamplingTimespan': 'adaptive_sampling_timespan', 'manualResume': 'manual_resume', 'timeUntilUp': 'time_until_up', 'upInterval': 'up_interval'}
    returnables = ['full_path', 'name', 'parent', 'description', 'adaptive', 'adaptive_divergence_type', 'adaptive_divergence_value', 'adaptive_limit', 'adaptive_sampling_timespan', 'destination', 'interval', 'manual_resume', 'time_until_up', 'timeout', 'transparent', 'up_interval']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def transparent(self):
        return flatten_boolean(self._values['transparent'])

    @property
    def manual_resume(self):
        return flatten_boolean(self._values['manual_resume'])

    @property
    def adaptive(self):
        return flatten_boolean(self._values['adaptive'])