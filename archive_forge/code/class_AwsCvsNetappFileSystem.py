from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
class AwsCvsNetappFileSystem(object):
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
        self.argument_spec.update(dict(state=dict(required=True, choices=['present', 'absent']), region=dict(required=True, type='str'), creationToken=dict(required=True, type='str', no_log=False), quotaInBytes=dict(required=False, type='int'), serviceLevel=dict(required=False, choices=['standard', 'premium', 'extreme']), exportPolicy=dict(type='dict', options=dict(rules=dict(type='list', elements='dict', options=dict(allowedClients=dict(required=False, type='str'), cifs=dict(required=False, type='bool'), nfsv3=dict(required=False, type='bool'), nfsv4=dict(required=False, type='bool'), ruleIndex=dict(required=False, type='int'), unixReadOnly=dict(required=False, type='bool'), unixReadWrite=dict(required=False, type='bool')))))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['region', 'creationToken', 'quotaInBytes'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = AwsCvsRestAPI(self.module)
        self.data = {}
        for key in self.parameters.keys():
            self.data[key] = self.parameters[key]

    def get_filesystem_id(self):
        list_filesystem, error = self.rest_api.get('FileSystems')
        if error:
            self.module.fail_json(msg=error)
        for filesystem in list_filesystem:
            if filesystem['creationToken'] == self.parameters['creationToken']:
                return filesystem['fileSystemId']
        return None

    def get_filesystem(self, filesystem_id):
        filesystem_info, error = self.rest_api.get('FileSystems/%s' % filesystem_id)
        if error:
            self.module.fail_json(msg=error)
        else:
            return filesystem_info
        return None

    def is_job_done(self, response):
        try:
            job_id = response['jobs'][0]['jobId']
        except TypeError:
            job_id = None
        if job_id is not None and self.rest_api.get_state(job_id) == 'done':
            return True
        return False

    def create_filesystem(self):
        api = 'FileSystems'
        response, error = self.rest_api.post(api, self.data)
        if not error:
            if self.is_job_done(response):
                return
            error = 'Error: unexpected response on FileSystems create: %s' % str(response)
        self.module.fail_json(msg=error)

    def delete_filesystem(self, filesystem_id):
        api = 'FileSystems/' + filesystem_id
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if not error:
            if self.is_job_done(response):
                return
            error = 'Error: unexpected response on FileSystems delete: %s' % str(response)
        self.module.fail_json(msg=error)

    def update_filesystem(self, filesystem_id):
        api = 'FileSystems/' + filesystem_id
        response, error = self.rest_api.put(api, self.data)
        if not error:
            if self.is_job_done(response):
                return
            error = 'Error: unexpected response on FileSystems update: %s' % str(response)
        self.module.fail_json(msg=error)

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        filesystem = None
        filesystem_id = self.get_filesystem_id()
        if filesystem_id:
            filesystem = self.get_filesystem(filesystem_id)
        cd_action = self.na_helper.get_cd_action(filesystem, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update_filesystem = False
            if filesystem['quotaInBytes'] is not None and 'quotaInBytes' in self.parameters and (filesystem['quotaInBytes'] != self.parameters['quotaInBytes']):
                update_filesystem = True
            elif filesystem['creationToken'] is not None and 'creationToken' in self.parameters and (filesystem['creationToken'] != self.parameters['creationToken']):
                update_filesystem = True
            elif filesystem['serviceLevel'] is not None and 'serviceLevel' in self.parameters and (filesystem['serviceLevel'] != self.parameters['serviceLevel']):
                update_filesystem = True
            elif 'exportPolicy' in filesystem and filesystem['exportPolicy']['rules'] is not None and ('exportPolicy' in self.parameters):
                for rule_org in filesystem['exportPolicy']['rules']:
                    for rule in self.parameters['exportPolicy']['rules']:
                        if rule_org['allowedClients'] != rule['allowedClients']:
                            update_filesystem = True
                        elif rule_org['unixReadOnly'] != rule['unixReadOnly']:
                            update_filesystem = True
                        elif rule_org['unixReadWrite'] != rule['unixReadWrite']:
                            update_filesystem = True
            if update_filesystem:
                self.na_helper.changed = True
        result_message = ''
        if self.na_helper.changed:
            if self.module.check_mode:
                result_message = 'Check mode, skipping changes'
            elif cd_action == 'create':
                self.create_filesystem()
                result_message = 'FileSystem Created'
            elif cd_action == 'delete':
                self.delete_filesystem(filesystem_id)
                result_message = 'FileSystem Deleted'
            else:
                self.update_filesystem(filesystem_id)
                result_message = 'FileSystem Updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message)