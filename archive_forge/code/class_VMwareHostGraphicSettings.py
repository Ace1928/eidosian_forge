from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VMwareHostGraphicSettings(PyVmomi):
    """ Main class for configuring Host Graphics Settings """

    def __init__(self, module):
        super(VMwareHostGraphicSettings, self).__init__(module)
        self.graphic_type = self.params['graphic_type']
        self.assigment_policy = self.params['assigment_policy']
        self.restart_xorg = self.params['restart_xorg']
        self.results = {'changed': False}
        esxi_hostname = self.params['esxi_hostname']
        cluster_name = self.params['cluster_name']
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_hostname)
        if self.hosts is None:
            self.module.fail_json(msg='Failed to find host system.')

    def ensure(self):
        """
        Function to manage host graphics
        """
        for host in self.hosts:
            self.results[host.name] = dict()
            if host.runtime.connectionState == 'connected':
                hgm = host.configManager.graphicsManager
                hsm = host.configManager.serviceSystem
                changed = False
                current_config = hgm.graphicsConfig
                if current_config.hostDefaultGraphicsType != self.graphic_type:
                    changed = True
                    current_config.hostDefaultGraphicsType = self.graphic_type
                if current_config.sharedPassthruAssignmentPolicy != self.assigment_policy:
                    changed = True
                    current_config.sharedPassthruAssignmentPolicy = self.assigment_policy
                if changed:
                    if self.module.check_mode:
                        not_world = '' if self.restart_xorg else 'not '
                        self.results[host.name]['changed'] = False
                        self.results[host.name]['msg'] = f"New host graphics settings will be changed to: hostDefaultGraphicsType =                                                             '{current_config.hostDefaultGraphicsType}', sharedPassthruAssignmentPolicy =                                                             '{current_config.sharedPassthruAssignmentPolicy}'.                                                             X.Org will {not_world}be restarted."
                    else:
                        try:
                            hgm.UpdateGraphicsConfig(current_config)
                            if self.restart_xorg:
                                hsm.RestartService('xorg')
                            xorg_status = 'was restarted' if self.restart_xorg else 'was not restarted.'
                            self.results['changed'] = True
                            self.results[host.name]['changed'] = True
                            self.results[host.name]['msg'] = f"New host graphics settings changed to: hostDefaultGraphicsType =                                                                 '{current_config.hostDefaultGraphicsType}', sharedPassthruAssignmentPolicy =                                                                 '{current_config.sharedPassthruAssignmentPolicy}'.                                                                 X.Org {xorg_status}"
                        except vim.fault.HostConfigFault as config_fault:
                            self.module.fail_json(msg=f'Failed to configure host graphics settings for host {host.name} due to : {to_native(config_fault.msg)}')
                else:
                    self.results[host.name]['changed'] = False
                    self.results[host.name]['msg'] = 'All Host Graphics Settings have already been configured'
            else:
                self.results[host.name]['changed'] = False
                self.results[host.name]['msg'] = f'Host {host.name} is disconnected and cannot be changed'
        self.module.exit_json(**self.results)