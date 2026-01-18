from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def ExitMaintenanceMode(self):
    if not self.host.runtime.inMaintenanceMode:
        self.module.exit_json(changed=False, hostsystem=str(self.host), hostname=self.esxi_hostname, status='NO_ACTION', msg='Host %s not in maintenance mode' % self.esxi_hostname)
    try:
        if not self.module.check_mode:
            task = self.host.ExitMaintenanceMode_Task(self.module.params['timeout'])
            success, result = wait_for_task(task)
        else:
            success = True
        self.module.exit_json(changed=success, hostsystem=str(self.host), hostname=self.esxi_hostname, status='EXIT', msg='Host %s exited maintenance mode' % self.esxi_hostname)
    except TaskError as e:
        self.module.fail_json(msg='Host %s failed to exit maintenance mode due to %s' % (self.esxi_hostname, to_native(e)))