from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _set_any_on_interface(self, ip='ipv6'):
    if ip == 'ipv4':
        self.want.update({'tunnel_local_address': 'any'})
        self.want.update({'tunnel_remote_address': 'any'})
    else:
        self.want.update({'tunnel_local_address': 'any6'})
        self.want.update({'tunnel_remote_address': 'any6'})