from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _check_replication_configured(module, blade):
    interfaces = blade.network_interfaces.list_network_interfaces()
    repl_ok = False
    for link in range(0, len(interfaces.items)):
        if 'replication' in interfaces.items[link].services:
            repl_ok = True
    if not repl_ok:
        module.fail_json(msg='Replication network interface required to configure a target')