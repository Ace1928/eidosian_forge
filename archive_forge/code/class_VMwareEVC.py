from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VMwareEVC(PyVmomi):

    def __init__(self, module):
        super(VMwareEVC, self).__init__(module)
        self.cluster_name = module.params['cluster_name']
        self.evc_mode = module.params['evc_mode']
        self.datacenter_name = module.params['datacenter_name']
        self.desired_state = module.params['state']
        self.datacenter = None
        self.cluster = None

    def process_state(self):
        """
        Manage internal states of evc
        """
        evc_states = {'absent': {'present': self.state_disable_evc, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_update_evc, 'absent': self.state_enable_evc}}
        current_state = self.check_evc_configuration()
        evc_states[self.desired_state][current_state]()

    def check_evc_configuration(self):
        """
        Check evc configuration
        Returns: 'Present' if evc enabled, else 'absent'
        """
        try:
            self.datacenter = find_datacenter_by_name(self.content, self.datacenter_name)
            if self.datacenter is None:
                self.module.fail_json(msg="Datacenter '%s' does not exist." % self.datacenter_name)
            self.cluster = self.find_cluster_by_name(cluster_name=self.cluster_name, datacenter_name=self.datacenter)
            if self.cluster is None:
                self.module.fail_json(msg="Cluster '%s' does not exist." % self.cluster_name)
            self.evcm = self.cluster.EvcManager()
            if not self.evcm:
                self.module.fail_json(msg="Unable to get EVC manager for cluster '%s'." % self.cluster_name)
            self.evc_state = self.evcm.evcState
            self.current_evc_mode = self.evc_state.currentEVCModeKey
            if not self.current_evc_mode:
                return 'absent'
            return 'present'
        except Exception as generic_exc:
            self.module.fail_json(msg='Failed to check configuration due to generic exception %s' % to_native(generic_exc))

    def state_exit_unchanged(self):
        """
        Exit without any change
        """
        self.module.exit_json(changed=False, msg="EVC Mode is already disabled on cluster '%s'." % self.cluster_name)

    def state_update_evc(self):
        """
        Update EVC Mode
        """
        changed, result = (False, None)
        try:
            if not self.module.check_mode and self.current_evc_mode != self.evc_mode:
                evc_task = self.evcm.ConfigureEvcMode_Task(self.evc_mode)
                changed, result = wait_for_task(evc_task)
            if self.module.check_mode and self.current_evc_mode != self.evc_mode:
                changed = True
            if self.current_evc_mode == self.evc_mode:
                self.module.exit_json(changed=changed, msg="EVC Mode is already set to '%(evc_mode)s' on '%(cluster_name)s'." % self.params)
            self.module.exit_json(changed=changed, msg="EVC Mode has been updated to '%(evc_mode)s' on '%(cluster_name)s'." % self.params)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to update EVC mode: %s' % to_native(invalid_argument))

    def state_enable_evc(self):
        """
        Enable EVC Mode
        """
        changed, result = (False, None)
        try:
            if not self.module.check_mode:
                evc_task = self.evcm.ConfigureEvcMode_Task(self.evc_mode)
                changed, result = wait_for_task(evc_task)
            if self.module.check_mode:
                changed = True
            self.module.exit_json(changed=changed, msg="EVC Mode for '%(evc_mode)s' has been enabled on '%(cluster_name)s'." % self.params)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to enable EVC mode: %s' % to_native(invalid_argument))

    def state_disable_evc(self):
        """
        Disable EVC Mode
        """
        changed, result = (False, None)
        try:
            if not self.module.check_mode:
                evc_task = self.evcm.DisableEvcMode_Task()
                changed, result = wait_for_task(evc_task)
            if self.module.check_mode:
                changed = True
            self.module.exit_json(changed=changed, msg="EVC Mode has been disabled on cluster '%s'." % self.cluster_name)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to disable EVC mode: %s' % to_native(invalid_argument))