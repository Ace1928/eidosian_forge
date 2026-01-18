from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPRestAPI(object):
    """ calls a REST API command """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(api=dict(required=True, type='str'), method=dict(required=False, type='str', default='GET'), query=dict(required=False, type='dict'), body=dict(required=False, type='dict', aliases=['info']), vserver_name=dict(required=False, type='str'), vserver_uuid=dict(required=False, type='str'), hal_linking=dict(required=False, type='bool', default=False), wait_for_completion=dict(required=False, type='bool', default=False), files=dict(required=False, type='dict'), accept_header=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        parameters = self.module.params
        self.api = parameters['api']
        self.method = parameters['method']
        self.query = parameters['query']
        self.body = parameters['body']
        self.vserver_name = parameters['vserver_name']
        self.vserver_uuid = parameters['vserver_uuid']
        self.hal_linking = parameters['hal_linking']
        self.wait_for_completion = parameters['wait_for_completion']
        self.files = parameters['files']
        self.accept_header = parameters['accept_header']
        self.rest_api = OntapRestAPI(self.module)
        if self.accept_header is None:
            self.accept_header = 'application/hal+json' if self.hal_linking else 'application/json'

    def build_headers(self):
        return self.rest_api.build_headers(accept=self.accept_header, vserver_name=self.vserver_name, vserver_uuid=self.vserver_uuid)

    def fail_on_error(self, status, response, error):
        if error:
            if isinstance(error, dict):
                error_message = error.pop('message', None)
                error_code = error.pop('code', None)
                if not error:
                    error = 'check error_message and error_code for details.'
            else:
                error_message = error
                error_code = None
            msg = "Error when calling '%s': %s" % (self.api, str(error))
            self.module.fail_json(msg=msg, status_code=status, response=response, error_message=error_message, error_code=error_code)

    def run_api(self):
        """ calls the REST API """
        status, response, error = self.rest_api.send_request(self.method, self.api, self.query, self.body, self.build_headers(), self.files)
        self.fail_on_error(status, response, error)
        return (status, response)

    def run_api_async(self):
        """ calls the REST API """
        args = [self.rest_api, self.api]
        kwargs = {}
        if self.method.upper() == 'POST':
            method = rest_generic.post_async
            kwargs['body'] = self.body
            kwargs['files'] = self.files
        elif self.method.upper() == 'PATCH':
            method = rest_generic.patch_async
            args.append(None)
            kwargs['body'] = self.body
            kwargs['files'] = self.files
        elif self.method.upper() == 'DELETE':
            method = rest_generic.delete_async
            args.append(None)
        else:
            self.module.warn('wait_for_completion ignored for %s method.' % self.method)
            return self.run_api()
        kwargs.update({'raw_error': True, 'headers': self.build_headers()})
        if self.query:
            kwargs['query'] = self.query
        response, error = method(*args, **kwargs)
        self.fail_on_error(0, response, error)
        return (0, response)

    def apply(self):
        """ calls the api and returns json output """
        changed_status = False if self.method.upper() == 'GET' else True
        if self.module.check_mode:
            status_code, response = (None, {'check_mode': 'would run %s %s' % (self.method, self.api)})
        elif self.wait_for_completion:
            status_code, response = self.run_api_async()
        else:
            status_code, response = self.run_api()
        self.module.exit_json(changed=changed_status, status_code=status_code, response=response)