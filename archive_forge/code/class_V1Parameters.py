from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class V1Parameters(Parameters):
    updatables = ['snmp_version', 'community', 'destination', 'port']
    returnables = ['snmp_version', 'community', 'destination', 'port']
    api_attributes = ['version', 'community', 'host', 'port']

    @property
    def network(self):
        return None