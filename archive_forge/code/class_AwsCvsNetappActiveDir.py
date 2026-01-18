from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
class AwsCvsNetappActiveDir(object):
    """
    Contains methods to parse arguments,
    derive details of AWS_CVS objects
    and send requests to AWS CVS via
    the restApi
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check paramenters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.aws_cvs_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=True, choices=['present', 'absent'], type='str'), region=dict(required=True, type='str'), DNS=dict(required=False, type='str'), domain=dict(required=False, type='str'), password=dict(required=False, type='str', no_log=True), netBIOS=dict(required=False, type='str'), username=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['domain', 'password'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = AwsCvsRestAPI(self.module)

    def get_activedirectory_id(self):
        try:
            list_activedirectory, error = self.rest_api.get('Storage/ActiveDirectory')
        except Exception:
            return None
        if error is not None:
            self.module.fail_json(msg='Error calling list_activedirectory: %s' % error)
        for activedirectory in list_activedirectory:
            if activedirectory['region'] == self.parameters['region']:
                return activedirectory['UUID']
        return None

    def get_activedirectory(self, activedirectory_id=None):
        if activedirectory_id is None:
            return None
        else:
            activedirectory_info, error = self.rest_api.get('Storage/ActiveDirectory/%s' % activedirectory_id)
            if not error:
                return activedirectory_info
            return None

    def create_activedirectory(self):
        api = 'Storage/ActiveDirectory'
        data = {'region': self.parameters['region'], 'DNS': self.parameters['DNS'], 'domain': self.parameters['domain'], 'username': self.parameters['username'], 'password': self.parameters['password'], 'netBIOS': self.parameters['netBIOS']}
        response, error = self.rest_api.post(api, data)
        if not error:
            return response
        else:
            self.module.fail_json(msg=response['message'])

    def delete_activedirectory(self):
        activedirectory_id = self.get_activedirectory_id()
        if activedirectory_id:
            api = 'Storage/ActiveDirectory/' + activedirectory_id
            data = None
            response, error = self.rest_api.delete(api, data)
            if not error:
                return response
            else:
                self.module.fail_json(msg=response['message'])
        else:
            self.module.fail_json(msg='Active Directory does not exist')

    def update_activedirectory(self, activedirectory_id, updated_activedirectory):
        api = 'Storage/ActiveDirectory/' + activedirectory_id
        data = {'region': self.parameters['region'], 'DNS': updated_activedirectory['DNS'], 'domain': updated_activedirectory['domain'], 'username': updated_activedirectory['username'], 'password': updated_activedirectory['password'], 'netBIOS': updated_activedirectory['netBIOS']}
        response, error = self.rest_api.put(api, data)
        if not error:
            return response
        else:
            self.module.fail_json(msg=response['message'])

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        modify = False
        activedirectory_id = self.get_activedirectory_id()
        current = self.get_activedirectory(activedirectory_id)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if current and self.parameters['state'] != 'absent':
            keys_to_check = ['DNS', 'domain', 'username', 'netBIOS']
            updated_active_directory, modify = self.na_helper.compare_and_update_values(current, self.parameters, keys_to_check)
            if self.parameters['password']:
                modify = True
                updated_active_directory['password'] = self.parameters['password']
            if modify is True:
                self.na_helper.changed = True
                if 'domain' in self.parameters and self.parameters['domain'] is not None:
                    ad_exists = self.get_activedirectory(updated_active_directory['domain'])
                    if ad_exists:
                        modify = False
                        self.na_helper.changed = False
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif modify is True:
                self.update_activedirectory(activedirectory_id, updated_active_directory)
            elif cd_action == 'create':
                self.create_activedirectory()
            elif cd_action == 'delete':
                self.delete_activedirectory()
        self.module.exit_json(changed=self.na_helper.changed)