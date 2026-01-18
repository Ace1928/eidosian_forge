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
def _validate_ports(self, item):
    start, stop = item.split('-')
    start = int(start.strip())
    stop = int(stop.strip())
    if 0 < start > 65535 or 0 < stop > 65535:
        raise F5ModuleError('Specified port number is out of valid range, correct range is between 0 and 65535.')
    return (start, stop)