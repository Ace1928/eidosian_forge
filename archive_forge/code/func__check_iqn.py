from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def _check_iqn(module, fusion):
    hap_api_instance = purefusion.HostAccessPoliciesApi(fusion)
    hosts = hap_api_instance.list_host_access_policies().items
    for host in hosts:
        if host.iqn == module.params['iqn'] and host.name != module.params['name']:
            module.fail_json(msg='Supplied IQN {0} already used by host access policy {1}'.format(module.params['iqn'], host.name))