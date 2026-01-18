from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
class VmwareHostHyperthreading(PyVmomi):
    """Manage Hyperthreading for an ESXi host system"""

    def __init__(self, module):
        super(VmwareHostHyperthreading, self).__init__(module)
        cluster_name = self.params.get('cluster_name')
        esxi_host_name = self.params.get('esxi_hostname')
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')

    def ensure(self):
        """Manage Hyperthreading for an ESXi host system"""
        results = dict(changed=False, result=dict())
        desired_state = self.params.get('state')
        host_change_list = []
        for host in self.hosts:
            changed = False
            results['result'][host.name] = dict(msg='')
            hyperthreading_info = host.config.hyperThread
            results['result'][host.name]['state'] = desired_state
            if desired_state == 'enabled':
                if hyperthreading_info.config:
                    if hyperthreading_info.active:
                        results['result'][host.name]['changed'] = False
                        results['result'][host.name]['state_current'] = 'active'
                        results['result'][host.name]['msg'] = 'Hyperthreading is enabled and active'
                    if not hyperthreading_info.active:
                        option_manager = host.configManager.advancedOption
                        try:
                            mitigation = option_manager.QueryOptions('VMkernel.Boot.hyperthreadingMitigation')
                        except vim.fault.InvalidName:
                            mitigation = None
                        if mitigation and mitigation[0].value:
                            results['result'][host.name]['changed'] = False
                            results['result'][host.name]['state_current'] = 'enabled'
                            results['result'][host.name]['msg'] = 'Hyperthreading is enabled, but not active because the processor is vulnerable to L1 Terminal Fault (L1TF).'
                        else:
                            changed = results['result'][host.name]['changed'] = True
                            results['result'][host.name]['state_current'] = 'enabled'
                            results['result'][host.name]['msg'] = 'Hyperthreading is enabled, but not active. A reboot is required!'
                elif hyperthreading_info.available:
                    if not self.module.check_mode:
                        try:
                            host.configManager.cpuScheduler.EnableHyperThreading()
                            changed = results['result'][host.name]['changed'] = True
                            results['result'][host.name]['state_previous'] = 'disabled'
                            results['result'][host.name]['state_current'] = 'enabled'
                            results['result'][host.name]['msg'] = 'Hyperthreading enabled for host. Reboot the host to activate it.'
                        except vmodl.fault.NotSupported as not_supported:
                            self.module.fail_json(msg="Failed to enable Hyperthreading for host '%s' : %s" % (host.name, to_native(not_supported.msg)))
                        except (vmodl.RuntimeFault, vmodl.MethodFault) as runtime_fault:
                            self.module.fail_json(msg="Failed to enable Hyperthreading for host '%s' due to : %s" % (host.name, to_native(runtime_fault.msg)))
                    else:
                        changed = results['result'][host.name]['changed'] = True
                        results['result'][host.name]['state_previous'] = 'disabled'
                        results['result'][host.name]['state_current'] = 'enabled'
                        results['result'][host.name]['msg'] = 'Hyperthreading will be enabled'
                else:
                    self.module.fail_json(msg="Hyperthreading optimization is not available for host '%s'" % host.name)
            elif desired_state == 'disabled':
                if not hyperthreading_info.config:
                    if not hyperthreading_info.active:
                        results['result'][host.name]['changed'] = False
                        results['result'][host.name]['state_current'] = 'inactive'
                        results['result'][host.name]['msg'] = 'Hyperthreading is disabled and inactive'
                    if hyperthreading_info.active:
                        changed = results['result'][host.name]['changed'] = True
                        results['result'][host.name]['state_current'] = 'disabled'
                        results['result'][host.name]['msg'] = 'Hyperthreading is already disabled but still active. A reboot is required!'
                elif hyperthreading_info.available:
                    if not self.module.check_mode:
                        try:
                            host.configManager.cpuScheduler.DisableHyperThreading()
                            changed = results['result'][host.name]['changed'] = True
                            results['result'][host.name]['state_previous'] = 'enabled'
                            results['result'][host.name]['state_current'] = 'disabled'
                            results['result'][host.name]['msg'] = 'Hyperthreading disabled. Reboot the host to deactivate it.'
                        except vmodl.fault.NotSupported as not_supported:
                            self.module.fail_json(msg="Failed to disable Hyperthreading for host '%s' : %s" % (host.name, to_native(not_supported.msg)))
                        except (vmodl.RuntimeFault, vmodl.MethodFault) as runtime_fault:
                            self.module.fail_json(msg="Failed to disable Hyperthreading for host '%s' due to : %s" % (host.name, to_native(runtime_fault.msg)))
                    else:
                        changed = results['result'][host.name]['changed'] = True
                        results['result'][host.name]['state_previous'] = 'enabled'
                        results['result'][host.name]['state_current'] = 'disabled'
                        results['result'][host.name]['msg'] = 'Hyperthreading will be disabled'
                else:
                    self.module.fail_json(msg="Hyperthreading optimization is not available for host '%s'" % host.name)
            host_change_list.append(changed)
        if any(host_change_list):
            results['changed'] = True
        self.module.exit_json(**results)