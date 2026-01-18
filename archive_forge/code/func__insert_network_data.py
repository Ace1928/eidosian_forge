from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _insert_network_data(server):
    for addr_data in server['ips']:
        if addr_data['type'] == 'IPV6':
            server['public_ipv6'] = addr_data['ip']
        elif addr_data['type'] == 'IPV4':
            server['public_ipv4'] = addr_data['ip']
    return server