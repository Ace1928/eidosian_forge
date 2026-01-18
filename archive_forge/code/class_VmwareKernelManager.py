from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VmwareKernelManager(PyVmomi):

    def __init__(self, module):
        self.module = module
        super(VmwareKernelManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        self.kernel_module_name = self.params.get('kernel_module_name')
        self.kernel_module_option = self.params.get('kernel_module_option')
        self.results = {}
        if not self.hosts:
            self.module.fail_json(msg='Failed to find a host system that matches the specified criteria')

    def get_kernel_module_option(self, host, kmod_name):
        host_kernel_manager = host.configManager.kernelModuleSystem
        try:
            return host_kernel_manager.QueryConfiguredModuleOptionString(self.kernel_module_name)
        except vim.fault.NotFound as kernel_fault:
            self.module.fail_json(msg="Failed to find kernel module on host '%s'. More information: %s" % (host.name, to_native(kernel_fault.msg)))

    def apply_kernel_module_option(self, host, kmod_name, kmod_option):
        host_kernel_manager = host.configManager.kernelModuleSystem
        if host_kernel_manager:
            try:
                if not self.module.check_mode:
                    host_kernel_manager.UpdateModuleOptionString(kmod_name, kmod_option)
            except vim.fault.NotFound as kernel_fault:
                self.module.fail_json(msg="Failed to find kernel module on host '%s'. More information: %s" % (host.name, to_native(kernel_fault)))
            except Exception as kernel_fault:
                self.module.fail_json(msg="Failed to configure kernel module for host '%s' due to: %s" % (host.name, to_native(kernel_fault)))

    def check_host_configuration_state(self):
        change_list = []
        for host in self.hosts:
            changed = False
            msg = ''
            self.results[host.name] = dict()
            if host.runtime.connectionState == 'connected':
                host_kernel_manager = host.configManager.kernelModuleSystem
                if host_kernel_manager:
                    original_options = self.get_kernel_module_option(host, self.kernel_module_name)
                    desired_options = self.kernel_module_option
                    if original_options != desired_options:
                        changed = True
                        if self.module.check_mode:
                            msg = 'Options would be changed on the kernel module'
                        else:
                            self.apply_kernel_module_option(host, self.kernel_module_name, desired_options)
                            msg = 'Options have been changed on the kernel module'
                            self.results[host.name]['configured_options'] = desired_options
                    else:
                        msg = 'Options are already the same'
                    change_list.append(changed)
                    self.results[host.name]['changed'] = changed
                    self.results[host.name]['msg'] = msg
                    self.results[host.name]['original_options'] = original_options
                else:
                    msg = 'No kernel module manager found on host %s - impossible to configure.' % host.name
                    self.results[host.name]['changed'] = changed
                    self.results[host.name]['msg'] = msg
            else:
                msg = 'Host %s is disconnected and cannot be changed.' % host.name
                self.results[host.name]['changed'] = changed
                self.results[host.name]['msg'] = msg
        self.module.exit_json(changed=any(change_list), host_kernel_status=self.results)