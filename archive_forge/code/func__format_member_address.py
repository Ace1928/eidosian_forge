from __future__ import absolute_import, division, print_function
import re
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _format_member_address(self, member):
    if len(member.split('%')) > 1:
        address, rd = member.split('%')
        if is_valid_ip(address):
            result = '/{0}/{1}%{2}'.format(self.partition, compress_address(address), rd)
            return result
    elif is_valid_ip(member):
        address = '/{0}/{1}'.format(self.partition, member)
        return address
    else:
        pattern = re.compile('(?!-)[A-Z-].*(?<!-)$', re.IGNORECASE)
        if pattern.match(member):
            address = '/{0}/{1}'.format(self.partition, member)
            return address
    raise F5ModuleError('The provided member address: {0} is not a valid IP address or snat translation name'.format(member))