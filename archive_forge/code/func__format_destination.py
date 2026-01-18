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
def _format_destination(self, address, port, route_domain):
    if port is None:
        if route_domain is None:
            result = '{0}'.format(fq_name(self.partition, address))
        else:
            result = '{0}%{1}'.format(fq_name(self.partition, address), route_domain)
    else:
        port = self._format_port_for_destination(address, port)
        if route_domain is None:
            result = '{0}{1}'.format(fq_name(self.partition, address), port)
        else:
            result = '{0}%{1}{2}'.format(fq_name(self.partition, address), route_domain, port)
    return result