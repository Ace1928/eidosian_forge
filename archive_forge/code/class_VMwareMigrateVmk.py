from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareMigrateVmk(object):

    def __init__(self, module):
        self.module = module
        self.host_system = None
        self.migrate_switch_name = self.module.params['migrate_switch_name']
        self.migrate_portgroup_name = self.module.params['migrate_portgroup_name']
        self.migrate_vlan_id = self.module.params['migrate_vlan_id']
        self.device = self.module.params['device']
        self.esxi_hostname = self.module.params['esxi_hostname']
        self.current_portgroup_name = self.module.params['current_portgroup_name']
        self.current_switch_name = self.module.params['current_switch_name']
        self.content = connect_to_api(module)

    def process_state(self):
        try:
            vmk_migration_states = {'migrate_vss_vds': self.state_migrate_vss_vds, 'migrate_vds_vss': self.state_migrate_vds_vss, 'migrated': self.state_exit_unchanged}
            vmk_migration_states[self.check_vmk_current_state()]()
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=runtime_fault.msg)
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=method_fault.msg)
        except Exception as e:
            self.module.fail_json(msg=str(e))

    def state_exit_unchanged(self):
        self.module.exit_json(changed=False)

    def create_host_vnic_config_vds_vss(self):
        host_vnic_config = vim.host.VirtualNic.Config()
        host_vnic_config.spec = vim.host.VirtualNic.Specification()
        host_vnic_config.changeOperation = 'edit'
        host_vnic_config.device = self.device
        host_vnic_config.spec.portgroup = self.migrate_portgroup_name
        return host_vnic_config

    def create_port_group_config_vds_vss(self):
        port_group_config = vim.host.PortGroup.Config()
        port_group_config.spec = vim.host.PortGroup.Specification()
        port_group_config.changeOperation = 'add'
        port_group_config.spec.name = self.migrate_portgroup_name
        port_group_config.spec.vlanId = self.migrate_vlan_id if self.migrate_vlan_id is not None else 0
        port_group_config.spec.vswitchName = self.migrate_switch_name
        port_group_config.spec.policy = vim.host.NetworkPolicy()
        return port_group_config

    def state_migrate_vds_vss(self):
        host_network_system = self.host_system.configManager.networkSystem
        config = vim.host.NetworkConfig()
        config.portgroup = [self.create_port_group_config_vds_vss()]
        host_network_system.UpdateNetworkConfig(config, 'modify')
        config = vim.host.NetworkConfig()
        config.vnic = [self.create_host_vnic_config_vds_vss()]
        host_network_system.UpdateNetworkConfig(config, 'modify')
        self.module.exit_json(changed=True)

    def create_host_vnic_config(self, dv_switch_uuid, portgroup_key):
        host_vnic_config = vim.host.VirtualNic.Config()
        host_vnic_config.spec = vim.host.VirtualNic.Specification()
        host_vnic_config.changeOperation = 'edit'
        host_vnic_config.device = self.device
        host_vnic_config.portgroup = ''
        host_vnic_config.spec.distributedVirtualPort = vim.dvs.PortConnection()
        host_vnic_config.spec.distributedVirtualPort.switchUuid = dv_switch_uuid
        host_vnic_config.spec.distributedVirtualPort.portgroupKey = portgroup_key
        return host_vnic_config

    def create_port_group_config(self):
        port_group_config = vim.host.PortGroup.Config()
        port_group_config.spec = vim.host.PortGroup.Specification()
        port_group_config.changeOperation = 'remove'
        port_group_config.spec.name = self.current_portgroup_name
        port_group_config.spec.vlanId = -1
        port_group_config.spec.vswitchName = self.current_switch_name
        port_group_config.spec.policy = vim.host.NetworkPolicy()
        return port_group_config

    def state_migrate_vss_vds(self):
        host_network_system = self.host_system.configManager.networkSystem
        dv_switch = find_dvs_by_name(self.content, self.migrate_switch_name)
        pg = find_dvspg_by_name(dv_switch, self.migrate_portgroup_name)
        config = vim.host.NetworkConfig()
        config.portgroup = [self.create_port_group_config()]
        config.vnic = [self.create_host_vnic_config(dv_switch.uuid, pg.key)]
        host_network_system.UpdateNetworkConfig(config, 'modify')
        self.module.exit_json(changed=True)

    def check_vmk_current_state(self):
        self.host_system = find_hostsystem_by_name(self.content, self.esxi_hostname)
        for vnic in self.host_system.configManager.networkSystem.networkInfo.vnic:
            if vnic.device == self.device:
                if vnic.spec.distributedVirtualPort is None:
                    std_vswitches = [vswitch.name for vswitch in self.host_system.configManager.networkSystem.networkInfo.vswitch]
                    if self.current_switch_name not in std_vswitches:
                        return 'migrated'
                    if vnic.portgroup == self.current_portgroup_name:
                        return 'migrate_vss_vds'
                else:
                    dvs = find_dvs_by_name(self.content, self.current_switch_name)
                    if dvs is None:
                        return 'migrated'
                    if vnic.spec.distributedVirtualPort.switchUuid == dvs.uuid:
                        return 'migrate_vds_vss'
        self.module.fail_json(msg='Unable to find the specified device %s.' % self.device)