from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _ip_data(self, ip_list):
    data = []
    for ip in list(ip_list):
        data.append({'address': ip.address, 'subnet_mask': ip.subnet_mask, 'gateway': ip.gateway, 'public': ip.public, 'prefix': ip.prefix, 'rdns': ip.rdns, 'type': ip.type})
    return data