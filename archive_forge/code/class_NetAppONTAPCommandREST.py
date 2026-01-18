from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
class NetAppONTAPCommandREST:
    """ calls a CLI command """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(command=dict(required=True, type='str'), verb=dict(required=True, type='str', choices=['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS']), params=dict(required=False, type='dict'), body=dict(required=False, type='dict')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.rest_api = OntapRestAPI(self.module)
        parameters = self.module.params
        self.command = parameters['command']
        self.verb = parameters['verb']
        self.params = parameters['params']
        self.body = parameters['body']
        if self.rest_api.is_rest():
            self.use_rest = True
        else:
            msg = 'failed to connect to REST over %s: %s' % (parameters['hostname'], self.rest_api.errors)
            msg += '.  Use na_ontap_command for non-rest CLI.'
            self.module.fail_json(msg=msg)

    def run_command(self):
        api = 'private/cli/' + self.command
        if self.verb == 'POST':
            message, error = self.rest_api.post(api, self.body, self.params)
        elif self.verb == 'GET':
            message, error = self.rest_api.get(api, self.params)
        elif self.verb == 'PATCH':
            message, error = self.rest_api.patch(api, self.body, self.params)
        elif self.verb == 'DELETE':
            message, error = self.rest_api.delete(api, self.body, self.params)
        elif self.verb == 'OPTIONS':
            message, error = self.rest_api.options(api, self.params)
        else:
            self.module.fail_json(msg='Error: unexpected verb %s' % self.verb, exception=traceback.format_exc())
        if error:
            self.module.fail_json(msg='Error: %s' % error)
        return message

    def apply(self):
        """ calls the command and returns raw output """
        changed = False if self.verb in ['GET', 'OPTIONS'] else True
        if self.module.check_mode:
            output = "Would run command: '%s'" % str(self.command)
        else:
            output = self.run_command()
        self.module.exit_json(changed=changed, msg=output)