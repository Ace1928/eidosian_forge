from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _convert_netmask(self, item):
    result = -1
    try:
        result = int(item)
        if 0 < result < 256:
            pass
    except ValueError:
        if is_valid_ip(item):
            ip = ip_network(u'0.0.0.0/%s' % str(item))
            result = ip.prefixlen
    if result < 0:
        raise F5ModuleError('The provided netmask {0} is neither in IP or CIDR format'.format(result))
    return result