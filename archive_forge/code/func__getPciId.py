from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
def _getPciId(self, host):
    for pnic in host.config.network.pnic:
        if pnic.device == self.vmnic:
            return pnic.pci
    self.module.fail_json(msg='No nic= %s on host= %s' % (self.vmnic, host.name))