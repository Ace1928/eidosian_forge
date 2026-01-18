from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def add_fdsp(self):
    """
        Adds a new File Directory Security Policy
        """
    api = 'private/cli/vserver/security/file-directory/policy'
    body = {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver']}
    dummy, error = self.rest_api.post(api, body)
    if error:
        self.module.fail_json(msg=error)