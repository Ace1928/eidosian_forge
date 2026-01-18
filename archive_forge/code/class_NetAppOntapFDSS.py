from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
class NetAppOntapFDSS:
    """
        Applys a File Directory Security Policy
    """

    def __init__(self):
        """
            Initialize the Ontap File Directory Security class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present'], default='present'), name=dict(required=True, type='str'), vserver=dict(required=True, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            self.module.fail_json(msg=self.rest_api.requires_ontap_version('na_ontap_fdss', '9.6'))

    def set_fdss(self):
        """
        Apply File Directory Security
        """
        api = 'private/cli/vserver/security/file-directory/apply'
        query = {'policy_name': self.parameters['name'], 'vserver': self.parameters['vserver']}
        response, error = self.rest_api.post(api, query)
        response, error = rrh.check_for_error_and_job_results(api, response, error, self.rest_api)
        if error:
            self.module.fail_json(msg=error)

    def apply(self):
        self.set_fdss()
        self.module.exit_json(changed=True)