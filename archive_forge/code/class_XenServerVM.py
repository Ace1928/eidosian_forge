from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.xenserver import (xenserver_common_argument_spec, XenServerObject, get_object_ref,
class XenServerVM(XenServerObject):
    """Class for managing XenServer VM.

    Attributes:
        vm_ref (str): XAPI reference to VM.
        vm_params (dict): A dictionary with VM parameters as returned
            by gather_vm_params() function.
    """

    def __init__(self, module):
        """Inits XenServerVM using module parameters.

        Args:
            module: Reference to Ansible module object.
        """
        super(XenServerVM, self).__init__(module)
        self.vm_ref = get_object_ref(self.module, self.module.params['name'], self.module.params['uuid'], obj_type='VM', fail=True, msg_prefix='VM search: ')
        self.gather_params()

    def gather_params(self):
        """Gathers all VM parameters available in XAPI database."""
        self.vm_params = gather_vm_params(self.module, self.vm_ref)

    def gather_facts(self):
        """Gathers and returns VM facts."""
        return gather_vm_facts(self.module, self.vm_params)

    def set_power_state(self, power_state):
        """Controls VM power state."""
        state_changed, current_state = set_vm_power_state(self.module, self.vm_ref, power_state, self.module.params['state_change_timeout'])
        if state_changed:
            self.vm_params['power_state'] = current_state.capitalize()
        return state_changed

    def wait_for_ip_address(self):
        """Waits for VM to acquire an IP address."""
        self.vm_params['guest_metrics'] = wait_for_vm_ip_address(self.module, self.vm_ref, self.module.params['state_change_timeout'])