from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VmwareServiceManager(PyVmomi):

    def __init__(self, module):
        super(VmwareServiceManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.options = self.params.get('options', dict())
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        self.desired_state = self.params.get('state')
        self.desired_policy = self.params.get('service_policy', None)
        self.service_name = self.params.get('service_name')
        self.results = {}

    def service_ctrl(self):
        changed = False
        host_service_state = []
        for host in self.hosts:
            actual_service_state, actual_service_policy = self.check_service_state(host=host, service_name=self.service_name)
            host_service_system = host.configManager.serviceSystem
            if host_service_system:
                changed_state = False
                self.results[host.name] = dict(service_name=self.service_name, actual_service_state='running' if actual_service_state else 'stopped', actual_service_policy=actual_service_policy, desired_service_policy=self.desired_policy, desired_service_state=self.desired_state, error='')
                try:
                    if self.desired_state in ['start', 'present']:
                        if not actual_service_state:
                            if not self.module.check_mode:
                                host_service_system.StartService(id=self.service_name)
                            changed_state = True
                    elif self.desired_state in ['stop', 'absent']:
                        if actual_service_state:
                            if not self.module.check_mode:
                                host_service_system.StopService(id=self.service_name)
                            changed_state = True
                    elif self.desired_state == 'restart':
                        if not self.module.check_mode:
                            host_service_system.RestartService(id=self.service_name)
                        changed_state = True
                    if self.desired_policy:
                        if actual_service_policy != self.desired_policy:
                            if not self.module.check_mode:
                                host_service_system.UpdateServicePolicy(id=self.service_name, policy=self.desired_policy)
                            changed_state = True
                    host_service_state.append(changed_state)
                    self.results[host.name].update(changed=changed_state)
                except (vim.fault.InvalidState, vim.fault.NotFound, vim.fault.HostConfigFault, vmodl.fault.InvalidArgument) as e:
                    self.results[host.name].update(changed=False, error=to_native(e.msg))
        if any(host_service_state):
            changed = True
        self.module.exit_json(changed=changed, host_service_status=self.results)

    def check_service_state(self, host, service_name):
        host_service_system = host.configManager.serviceSystem
        if host_service_system:
            services = host_service_system.serviceInfo.service
            for service in services:
                if service.key == service_name:
                    return (service.running, service.policy)
        msg = "Failed to find '%s' service on host system '%s'" % (service_name, host.name)
        cluster_name = self.params.get('cluster_name', None)
        if cluster_name:
            msg += " located on cluster '%s'" % cluster_name
        msg += ', please check if you have specified a valid ESXi service name.'
        self.module.fail_json(msg=msg)