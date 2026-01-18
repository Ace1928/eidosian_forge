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
class RouteDomainParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'bwcPolicy': 'bwc_policy', 'connectionLimit': 'connection_limit', 'flowEvictionPolicy': 'flow_eviction_policy', 'servicePolicy': 'service_policy', 'routingProtocol': 'routing_protocol'}
    returnables = ['name', 'id', 'full_path', 'parent', 'bwc_policy', 'connection_limit', 'description', 'flow_eviction_policy', 'service_policy', 'strict', 'routing_protocol', 'vlans']

    @property
    def strict(self):
        return flatten_boolean(self._values['strict'])

    @property
    def connection_limit(self):
        if self._values['connection_limit'] is None:
            return None
        return int(self._values['connection_limit'])

    @property
    def id(self):
        if self._values['id'] is None:
            return None
        return int(self._values['id'])