from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
class VmwareHostPowerManagement(PyVmomi):
    """
    Class to manage power management policy of an ESXi host system
    """

    def __init__(self, module):
        super(VmwareHostPowerManagement, self).__init__(module)
        cluster_name = self.params.get('cluster_name')
        esxi_host_name = self.params.get('esxi_hostname')
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system with given configuration.')

    def ensure(self):
        """
        Manage power management policy of an ESXi host system
        """
        results = dict(changed=False, result=dict())
        policy = self.params.get('policy')
        host_change_list = []
        power_policies = {'high-performance': {'key': 1, 'short_name': 'static'}, 'balanced': {'key': 2, 'short_name': 'dynamic'}, 'low-power': {'key': 3, 'short_name': 'low'}, 'custom': {'key': 4, 'short_name': 'custom'}}
        for host in self.hosts:
            changed = False
            results['result'][host.name] = dict(msg='')
            power_system = host.configManager.powerSystem
            power_system_info = power_system.info
            current_host_power_policy = power_system_info.currentPolicy
            if current_host_power_policy.shortName == 'static':
                current_policy = 'high-performance'
            elif current_host_power_policy.shortName == 'dynamic':
                current_policy = 'balanced'
            elif current_host_power_policy.shortName == 'low':
                current_policy = 'low-power'
            elif current_host_power_policy.shortName == 'custom':
                current_policy = 'custom'
            results['result'][host.name]['desired_state'] = policy
            if current_host_power_policy.key == power_policies[policy]['key']:
                results['result'][host.name]['changed'] = changed
                results['result'][host.name]['previous_state'] = current_policy
                results['result'][host.name]['current_state'] = policy
                results['result'][host.name]['msg'] = 'Power policy is already configured'
            else:
                supported_policy = False
                power_system_capability = power_system.capability
                available_host_power_policies = power_system_capability.availablePolicy
                for available_policy in available_host_power_policies:
                    if available_policy.shortName == power_policies[policy]['short_name']:
                        supported_policy = True
                if supported_policy:
                    if not self.module.check_mode:
                        try:
                            power_system.ConfigurePowerPolicy(key=power_policies[policy]['key'])
                            changed = True
                            results['result'][host.name]['changed'] = True
                            results['result'][host.name]['msg'] = 'Power policy changed'
                        except vmodl.fault.InvalidArgument:
                            self.module.fail_json(msg="Invalid power policy key provided for host '%s'" % host.name)
                        except vim.fault.HostConfigFault as host_config_fault:
                            self.module.fail_json(msg="Failed to configure power policy for host '%s': %s" % (host.name, to_native(host_config_fault.msg)))
                    else:
                        changed = True
                        results['result'][host.name]['changed'] = True
                        results['result'][host.name]['msg'] = 'Power policy will be changed'
                    results['result'][host.name]['previous_state'] = current_policy
                    results['result'][host.name]['current_state'] = policy
                else:
                    changed = False
                    results['result'][host.name]['changed'] = changed
                    results['result'][host.name]['previous_state'] = current_policy
                    results['result'][host.name]['current_state'] = current_policy
                    self.module.fail_json(msg="Power policy '%s' isn't supported for host '%s'" % (policy, host.name))
            host_change_list.append(changed)
        if any(host_change_list):
            results['changed'] = True
        self.module.exit_json(**results)