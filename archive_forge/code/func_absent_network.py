from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_network(self):
    physical_network = self.get_physical_network()
    if physical_network:
        self.result['changed'] = True
        args = {'id': physical_network['id']}
        if not self.module.check_mode:
            resource = self.query_api('deletePhysicalNetwork', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(resource, 'success')
    return physical_network