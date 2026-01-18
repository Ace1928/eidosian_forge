from __future__ import absolute_import, division, print_function
import os
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip, validate_ip_v6_address
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _join_address_port(self, item):
    if 'port' not in item:
        raise F5ModuleError('Aggregates must be provided with both address and port.')
    delimiter = ':'
    if 'name' in item and item['name']:
        return '{0}{1}{2}'.format(item['name'], delimiter, item['port'])
    try:
        if validate_ip_v6_address(item['address']):
            delimiter = '.'
    except TypeError:
        pass
    return '{0}{1}{2}'.format(item['address'], delimiter, item['port'])