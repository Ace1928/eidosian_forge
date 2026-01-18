from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapSecuritySSH:
    """ object initialize and class methods """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), vserver=dict(required=False, type='str'), ciphers=dict(required=False, type='list', elements='str'), key_exchange_algorithms=dict(required=False, type='list', elements='str', no_log=False), mac_algorithms=dict(required=False, type='list', elements='str'), max_authentication_retry_count=dict(required=False, type='int')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule(self)
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.svm_uuid = None
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_security_ssh', 9, 10, 1)
        self.safe_strip()

    def safe_strip(self):
        """ strip the left and right spaces of string and also removes an empty string"""
        for option in ('ciphers', 'key_exchange_algorithms', 'mac_algorithms'):
            if option in self.parameters:
                self.parameters[option] = [item.strip() for item in self.parameters[option] if len(item.strip())]
                if self.parameters[option] == []:
                    self.module.fail_json(msg='Removing all SSH %s is not supported. SSH login would fail. There must be at least one %s associated with the SSH configuration.' % (option, option))
        return

    def get_security_ssh_rest(self):
        """
        Retrieves the SSH server configuration for the SVM or cluster.
        """
        fields = ['key_exchange_algorithms', 'ciphers', 'mac_algorithms', 'max_authentication_retry_count']
        query = {}
        if self.parameters.get('vserver'):
            api = 'security/ssh/svms'
            query['svm.name'] = self.parameters['vserver']
            fields.append('svm.uuid')
        else:
            api = 'security/ssh'
        query['fields'] = ','.join(fields)
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg=error)
        return record

    def modify_security_ssh_rest(self, modify):
        """
        Updates the SSH server configuration for the specified SVM.
        """
        if self.parameters.get('vserver'):
            if self.svm_uuid is None:
                self.module.fail_json(msg='Error: no uuid found for the SVM')
            api = 'security/ssh/svms'
        else:
            api = 'security/ssh'
        body = {}
        for option in ('ciphers', 'key_exchange_algorithms', 'mac_algorithms', 'max_authentication_retry_count'):
            if option in modify:
                body[option] = modify[option]
        if body:
            dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
            if error:
                self.module.fail_json(msg=error)

    def apply(self):
        current = self.get_security_ssh_rest()
        self.svm_uuid = self.na_helper.safe_get(current, ['svm', 'uuid']) if current and self.parameters.get('vserver') else None
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_security_ssh_rest(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, modify=modify)
        self.module.exit_json(**result)