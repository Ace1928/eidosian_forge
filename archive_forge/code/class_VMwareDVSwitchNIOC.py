from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareDVSwitchNIOC(PyVmomi):

    def __init__(self, module):
        super(VMwareDVSwitchNIOC, self).__init__(module)
        self.dvs = None
        self.resource_changes = list()
        self.switch = module.params['switch']
        self.version = module.params.get('version')
        self.state = module.params['state']
        self.resources = module.params.get('resources')
        self.result = {'changed': False, 'dvswitch_nioc_status': 'Unchanged', 'resources_changed': list()}

    def process_state(self):
        nioc_states = {'absent': {'present': self.state_disable_nioc, 'absent': self.state_exit}, 'present': {'version': self.state_update_nioc_version, 'update': self.state_update_nioc_resources, 'present': self.state_exit, 'absent': self.state_enable_nioc}}
        nioc_states[self.state][self.check_nioc_state()]()
        self.state_exit()

    def state_exit(self):
        self.module.exit_json(**self.result)

    def state_disable_nioc(self):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.set_nioc_enabled(False)
        self.result['dvswitch_nioc_status'] = 'Disabled NIOC'

    def state_enable_nioc(self):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.set_nioc_enabled(True)
            self.set_nioc_version()
            self.result['dvswitch_nioc_status'] = 'Enabled NIOC with version %s' % self.version
            if self.check_resources() == 'update':
                self.set_nioc_resources(self.resource_changes)

    def state_update_nioc_version(self):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.set_nioc_version()
            self.result['dvswitch_nioc_status'] = 'Set NIOC to version %s' % self.version
            if self.check_resources() == 'update':
                self.set_nioc_resources(self.resource_changes)

    def state_update_nioc_resources(self):
        self.result['changed'] = True
        if not self.module.check_mode:
            self.result['dvswitch_nioc_status'] = 'Resource configuration modified'
            self.set_nioc_resources(self.resource_changes)

    def set_nioc_enabled(self, state):
        try:
            self.dvs.EnableNetworkResourceManagement(enable=state)
        except vim.fault.DvsFault as dvs_fault:
            self.module.fail_json(msg='DvsFault while setting NIOC enabled=%r: %s' % (state, to_native(dvs_fault.msg)))
        except vim.fault.DvsNotAuthorized as auth_fault:
            self.module.fail_json(msg='Not authorized to set NIOC enabled=%r: %s' % (state, to_native(auth_fault.msg)))
        except vmodl.fault.NotSupported as support_fault:
            self.module.fail_json(msg='NIOC not supported by DVS: %s' % to_native(support_fault.msg))
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg='RuntimeFault while setting NIOC enabled=%r: %s' % (state, to_native(runtime_fault.msg)))

    def set_nioc_version(self):
        upgrade_spec = vim.DistributedVirtualSwitch.ConfigSpec()
        upgrade_spec.configVersion = self.dvs.config.configVersion
        if not self.version:
            self.version = 'version2'
        upgrade_spec.networkResourceControlVersion = self.version
        try:
            task = self.dvs.ReconfigureDvs_Task(spec=upgrade_spec)
            wait_for_task(task)
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg='RuntimeFault when setting NIOC version: %s ' % to_native(runtime_fault.msg))

    def check_nioc_state(self):
        self.dvs = find_dvs_by_name(self.content, self.switch)
        if self.dvs is None:
            self.module.fail_json(msg='DVS %s was not found.' % self.switch)
        else:
            if not self.dvs.config.networkResourceManagementEnabled:
                return 'absent'
            if self.version and self.dvs.config.networkResourceControlVersion != self.version:
                return 'version'
            return self.check_resources()

    def check_resources(self):
        self.dvs = find_dvs_by_name(self.content, self.switch)
        if self.dvs is None:
            self.module.fail_json(msg="DVS named '%s' was not found" % self.switch)
        for resource in self.resources:
            if self.check_resource_state(resource) == 'update':
                self.resource_changes.append(resource)
                self.result['resources_changed'].append(resource['name'])
        if len(self.resource_changes) > 0:
            return 'update'
        return 'present'

    def check_resource_state(self, resource):
        resource_cfg = self.find_netioc_by_key(resource['name'])
        if resource_cfg is None:
            self.module.fail_json(msg="NetIOC resource named '%s' was not found" % resource['name'])
        rc = {'limit': resource_cfg.allocationInfo.limit, 'shares_level': resource_cfg.allocationInfo.shares.level}
        if resource_cfg.allocationInfo.shares.level == 'custom':
            rc['shares'] = resource_cfg.allocationInfo.shares.shares
        if self.dvs.config.networkResourceControlVersion == 'version3':
            rc['reservation'] = resource_cfg.allocationInfo.reservation
        for k, v in rc.items():
            if k in resource and v != resource[k]:
                return 'update'
        return 'valid'

    def set_nioc_resources(self, resources):
        if self.dvs.config.networkResourceControlVersion == 'version3':
            self._update_version3_resources(resources)
        elif self.dvs.config.networkResourceControlVersion == 'version2':
            self._update_version2_resources(resources)

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

    def _update_version2_resources(self, resources):
        allocations = list()
        for resource in resources:
            resource_cfg = self.find_netioc_by_key(resource['name'])
            allocation = vim.DVSNetworkResourcePoolConfigSpec()
            allocation.allocationInfo = vim.DVSNetworkResourcePoolAllocationInfo()
            allocation.key = resource['name']
            allocation.configVersion = resource_cfg.configVersion
            if 'limit' in resource:
                allocation.allocationInfo.limit = resource['limit']
            if 'shares_level' in resource:
                allocation.allocationInfo.shares = vim.SharesInfo()
                allocation.allocationInfo.shares.level = resource['shares_level']
                if 'shares' in resource and resource['shares_level'] == 'custom':
                    allocation.allocationInfo.shares.shares = resource['shares']
            allocations.append(allocation)
        self.dvs.UpdateNetworkResourcePool(allocations)

    def find_netioc_by_key(self, resource_name):
        config = None
        if self.dvs.config.networkResourceControlVersion == 'version3':
            config = self.dvs.config.infrastructureTrafficResourceConfig
        elif self.dvs.config.networkResourceControlVersion == 'version2':
            config = self.dvs.networkResourcePool
        for obj in config:
            if obj.key == resource_name:
                return obj
        return None