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
def _read_current_message_routing_profiles_from_device(self):
    result = []
    result += self._read_diameter_profiles_from_device()
    result += self._read_sip_profiles_from_device()
    result += self._read_legacy_sip_profiles_from_device()
    return result