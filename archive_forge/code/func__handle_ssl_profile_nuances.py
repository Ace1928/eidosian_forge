from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _handle_ssl_profile_nuances(self, profile):
    if self.check_profiles:
        if profile['name'] == 'serverssl' or self._is_server_ssl_profile(profile):
            if profile['context'] != 'serverside':
                profile['context'] = 'serverside'
        if profile['name'] == 'clientssl' or self._is_client_ssl_profile(profile):
            if profile['context'] != 'clientside':
                profile['context'] = 'clientside'
    else:
        if profile['name'] == 'serverssl':
            if profile['context'] != 'serverside':
                profile['context'] = 'serverside'
        if profile['name'] == 'clientssl':
            if profile['context'] != 'clientside':
                profile['context'] = 'clientside'