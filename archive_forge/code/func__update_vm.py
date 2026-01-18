from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _update_vm(self, resource_group, name, params):
    try:
        poller = self.compute_client.virtual_machines.begin_create_or_update(resource_group, name, params)
        self.get_poller_result(poller)
    except Exception as exc:
        if self.managed_by_extended:
            return exc
        else:
            self.fail('Error updating virtual machine {0} - {1}'.format(name, str(exc)))