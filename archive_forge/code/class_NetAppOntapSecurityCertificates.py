from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
class NetAppOntapSecurityCertificates:
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(common_name=dict(required=False, type='str'), name=dict(required=False, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), type=dict(required=False, choices=['client', 'server', 'client_ca', 'server_ca', 'root_ca']), svm=dict(required=False, type='str', aliases=['vserver']), public_certificate=dict(required=False, type='str'), private_key=dict(required=False, type='str', no_log=True), signing_request=dict(required=False, type='str'), expiry_time=dict(required=False, type='str'), key_size=dict(required=False, type='int'), hash_function=dict(required=False, type='str'), intermediate_certificates=dict(required=False, type='list', elements='str'), ignore_name_if_not_supported=dict(required=False, type='bool', default=True)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if self.parameters.get('name') is None and (self.parameters.get('common_name') is None or self.parameters.get('type') is None):
            error = "Error: 'name' or ('common_name' and 'type') are required parameters."
            self.module.fail_json(msg=error)
        self.ignore_name_param = False
        self.rest_api = OntapRestAPI(self.module)
        if self.rest_api.is_rest():
            self.use_rest = True
        else:
            self.module.fail_json(msg=self.rest_api.requires_ontap_9_6('na_ontap_security_certificates'))

    def get_certificate(self):
        """
        Fetch uuid if certificate exists.
        NOTE: because of a bug in ONTAP 9.6 and 9.7, name is not supported. We are
        falling back to using common_name and type, but unicity is not guaranteed.
        :return:
            Dictionary if certificate with same name is found
            None if not found
        """
        if 'svm' in self.parameters:
            rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['svm'], self.module, True)
        error = "'name' or ('common_name', 'type') are required."
        for key in ('name', 'common_name'):
            if self.parameters.get(key) is None:
                continue
            data = {'fields': 'uuid', key: self.parameters[key]}
            if self.parameters.get('svm') is not None:
                data['svm.name'] = self.parameters['svm']
            else:
                data['scope'] = 'cluster'
            if key == 'common_name':
                if self.parameters.get('type') is not None:
                    data['type'] = self.parameters['type']
                else:
                    error = "When using 'common_name', 'type' is required."
                    break
            api = 'security/certificates'
            message, error = self.rest_api.get(api, data)
            if error:
                try:
                    name_not_supported_error = key == 'name' and error['message'] == 'Unexpected argument "name".'
                except (KeyError, TypeError):
                    name_not_supported_error = False
                if name_not_supported_error:
                    if self.parameters['ignore_name_if_not_supported'] and self.parameters.get('common_name') is not None:
                        self.ignore_name_param = True
                        continue
                    error = "ONTAP 9.6 and 9.7 do not support 'name'.  Use 'common_name' and 'type' as a work-around."
            break
        if error:
            self.module.fail_json(msg='Error calling API: %s - %s' % (api, error))
        if len(message['records']) == 1:
            return message['records'][0]
        if len(message['records']) > 1:
            error = 'Duplicate records with same common_name are preventing safe operations: %s' % repr(message)
            self.module.fail_json(msg=error)
        return None

    def create_or_install_certificate(self, validate_only=False):
        """
        Create or install certificate
        :return: message (should be empty dict)
        """
        required_keys = ['type', 'common_name']
        if validate_only:
            if not set(required_keys).issubset(set(self.parameters.keys())):
                self.module.fail_json(msg='Error creating or installing certificate: one or more of the following options are missing: %s' % ', '.join(required_keys))
            return
        optional_keys = ['public_certificate', 'private_key', 'expiry_time', 'key_size', 'hash_function', 'intermediate_certificates']
        if not self.ignore_name_param:
            optional_keys.append('name')
        body = {}
        if self.parameters.get('svm') is not None:
            body['svm'] = {'name': self.parameters['svm']}
        for key in required_keys + optional_keys:
            if self.parameters.get(key) is not None:
                body[key] = self.parameters[key]
        params = {'return_records': 'true'}
        api = 'security/certificates'
        message, error = self.rest_api.post(api, body, params)
        if error:
            if self.parameters.get('svm') is None and error.get('target') == 'uuid':
                error['target'] = 'cluster'
            if error.get('message') == 'duplicate entry':
                error['message'] += '.  Same certificate may already exist under a different name.'
            self.module.fail_json(msg='Error creating or installing certificate: %s' % error)
        return message

    def sign_certificate(self, uuid):
        """
        sign certificate
        :return: a dictionary with key "public_certificate"
        """
        api = 'security/certificates/%s/sign' % uuid
        body = {'signing_request': self.parameters['signing_request']}
        optional_keys = ['expiry_time', 'hash_function']
        for key in optional_keys:
            if self.parameters.get(key) is not None:
                body[key] = self.parameters[key]
        params = {'return_records': 'true'}
        message, error = self.rest_api.post(api, body, params)
        if error:
            self.module.fail_json(msg='Error signing certificate: %s' % error)
        return message

    def delete_certificate(self, uuid):
        """
        Delete certificate
        :return: message (should be empty dict)
        """
        api = 'security/certificates/%s' % uuid
        message, error = self.rest_api.delete(api)
        if error:
            self.module.fail_json(msg='Error deleting certificate: %s' % error)
        return message

    def apply(self):
        """
        Apply action to create/install/sign/delete certificate
        :return: None
        """
        current = self.get_certificate()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        message = None
        if self.parameters.get('signing_request') is not None:
            error = None
            if self.parameters['state'] == 'absent':
                error = "'signing_request' is not supported with 'state' set to 'absent'"
            elif current is None:
                scope = 'cluster' if self.parameters.get('svm') is None else 'svm: %s' % self.parameters.get('svm')
                error = "signing certificate with name '%s' not found on %s" % (self.parameters.get('name'), scope)
            elif cd_action is not None:
                error = "'signing_request' is exclusive with other actions: create, install, delete"
            if error is not None:
                self.module.fail_json(msg=error)
            cd_action = 'sign'
            self.na_helper.changed = True
        if self.na_helper.changed and cd_action == 'create':
            self.create_or_install_certificate(validate_only=True)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                message = self.create_or_install_certificate()
            elif cd_action == 'sign':
                message = self.sign_certificate(current['uuid'])
            elif cd_action == 'delete':
                message = self.delete_certificate(current['uuid'])
        results = netapp_utils.generate_result(self.na_helper.changed, cd_action, extra_responses={'ontap_info': message})
        self.module.exit_json(**results)