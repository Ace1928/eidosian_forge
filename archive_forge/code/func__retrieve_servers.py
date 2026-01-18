from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.common.text.converters import to_native
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from collections import namedtuple
import os
def _retrieve_servers(self, label_filter=None):
    vm_pool = self._get_vm_pool()
    result = []
    for vm in vm_pool.VM:
        server = vm.USER_TEMPLATE
        labels = []
        if vm.USER_TEMPLATE.get('LABELS'):
            labels = [s for s in vm.USER_TEMPLATE.get('LABELS') if s == ',' or s == '-' or s.isalnum() or s.isspace()]
            labels = ''.join(labels)
            labels = labels.replace(' ', '_')
            labels = labels.replace('-', '_')
            labels = labels.split(',')
        if label_filter is not None:
            if label_filter not in labels:
                continue
        server['name'] = vm.NAME
        server['LABELS'] = labels
        server['v4_first_ip'] = self._get_vm_ipv4(vm)
        server['v6_first_ip'] = self._get_vm_ipv6(vm)
        result.append(server)
    return result