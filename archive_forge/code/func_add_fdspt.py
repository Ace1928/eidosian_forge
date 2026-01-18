from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def add_fdspt(self):
    """
        Adds a new File Directory Security Policy Task
        """
    api = 'private/cli/vserver/security/file-directory/policy/task/add'
    body = {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver'], 'path': self.parameters['path']}
    for i in ('ntfs_mode', 'ntfs_sd', 'security_type', 'access_control', 'index_num'):
        if i in self.parameters:
            body[i.replace('_', '-')] = self.parameters[i]
    dummy, error = self.rest_api.post(api, body)
    if error:
        self.module.fail_json(msg=error)