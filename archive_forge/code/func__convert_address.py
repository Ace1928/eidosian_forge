from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import ip_network
from ..module_utils.teem import send_teem
def _convert_address(self, item):
    ip = ip_network(u'{0}'.format(item))
    return ip.with_prefixlen