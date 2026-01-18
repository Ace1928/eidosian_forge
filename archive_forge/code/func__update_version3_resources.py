from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _update_version3_resources(self, resources):
    allocations = list()
    for resource in resources:
        allocation = vim.DistributedVirtualSwitch.HostInfrastructureTrafficResource()
        allocation.allocationInfo = vim.DistributedVirtualSwitch.HostInfrastructureTrafficResource.ResourceAllocation()
        allocation.key = resource['name']
        if 'limit' in resource:
            allocation.allocationInfo.limit = resource['limit']
        if 'reservation' in resource:
            allocation.allocationInfo.reservation = resource['reservation']
        if 'shares_level' in resource:
            allocation.allocationInfo.shares = vim.SharesInfo()
            allocation.allocationInfo.shares.level = resource['shares_level']
            if 'shares' in resource and resource['shares_level'] == 'custom':
                allocation.allocationInfo.shares.shares = resource['shares']
            elif resource['shares_level'] == 'custom':
                self.module.fail_json(msg='Resource %s, shares_level set to custom but shares not specified' % resource['name'])
        allocations.append(allocation)
    spec = vim.DistributedVirtualSwitch.ConfigSpec()
    spec.configVersion = self.dvs.config.configVersion
    spec.infrastructureTrafficResourceConfig = allocations
    task = self.dvs.ReconfigureDvs_Task(spec)
    wait_for_task(task)