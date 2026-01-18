from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class VdirectRunnable(object):
    CREATE_WORKFLOW_ACTION = 'createWorkflow'
    RUN_ACTION = 'run'

    def __init__(self, params):
        self.client = rest_client.RestClient(params['vdirect_ip'], params['vdirect_user'], params['vdirect_password'], wait=params['vdirect_wait'], secondary_vdirect_ip=params['vdirect_secondary_ip'], https_port=params['vdirect_https_port'], http_port=params['vdirect_http_port'], timeout=params['vdirect_timeout'], strict_http_results=True, https=params['vdirect_use_ssl'], verify=params['validate_certs'])
        self.params = params
        self.type = self.params['runnable_type']
        self.name = self.params['runnable_name']
        if self.type == WORKFLOW_TEMPLATE_RUNNABLE_TYPE:
            self.action_name = VdirectRunnable.CREATE_WORKFLOW_ACTION
        elif self.type == CONFIGURATION_TEMPLATE_RUNNABLE_TYPE:
            self.action_name = VdirectRunnable.RUN_ACTION
        else:
            self.action_name = self.params['action_name']
        if 'parameters' in self.params and self.params['parameters']:
            self.action_params = self.params['parameters']
        else:
            self.action_params = {}

    def _validate_runnable_exists(self):
        if self.type == WORKFLOW_RUNNABLE_TYPE:
            res = self.client.runnable.get_runnable_objects(self.type)
            runnable_names = res[rest_client.RESP_DATA]['names']
            if self.name not in runnable_names:
                raise MissingRunnableException(self.name)
        else:
            try:
                self.client.catalog.get_catalog_item(self.type, self.name)
            except rest_client.RestClientException:
                raise MissingRunnableException(self.name)

    def _validate_action_name(self):
        if self.type in [WORKFLOW_RUNNABLE_TYPE, PLUGIN_RUNNABLE_TYPE]:
            res = self.client.runnable.get_available_actions(self.type, self.name)
            available_actions = res[rest_client.RESP_DATA]['names']
            if self.action_name not in available_actions:
                raise WrongActionNameException(self.action_name, available_actions)

    def _validate_required_action_params(self):
        action_params_names = list(self.action_params)
        res = self.client.runnable.get_action_info(self.type, self.name, self.action_name)
        if 'parameters' in res[rest_client.RESP_DATA]:
            action_params_spec = res[rest_client.RESP_DATA]['parameters']
        else:
            action_params_spec = []
        required_action_params_dict = [{'name': p['name'], 'type': p['type']} for p in action_params_spec if p['type'] == 'alteon' or p['type'] == 'defensePro' or p['type'] == 'appWall' or (p['type'] == 'alteon[]') or (p['type'] == 'defensePro[]') or (p['type'] == 'appWall[]') or (p['direction'] != 'out')]
        required_action_params_names = [n['name'] for n in required_action_params_dict]
        if set(required_action_params_names) & set(action_params_names) != set(required_action_params_names):
            raise MissingActionParametersException(required_action_params_dict)

    def run(self):
        self._validate_runnable_exists()
        self._validate_action_name()
        self._validate_required_action_params()
        data = self.action_params
        result = self.client.runnable.run(data, self.type, self.name, self.action_name)
        result_to_return = {'msg': ''}
        if result[rest_client.RESP_STATUS] == 200:
            if result[rest_client.RESP_DATA]['success']:
                if self.type == WORKFLOW_TEMPLATE_RUNNABLE_TYPE:
                    result_to_return['msg'] = WORKFLOW_CREATION_SUCCESS
                elif self.type == CONFIGURATION_TEMPLATE_RUNNABLE_TYPE:
                    result_to_return['msg'] = TEMPLATE_SUCCESS
                elif self.type == PLUGIN_RUNNABLE_TYPE:
                    result_to_return['msg'] = PLUGIN_ACTION_SUCCESS
                else:
                    result_to_return['msg'] = WORKFLOW_ACTION_SUCCESS
                result_to_return['output'] = result[rest_client.RESP_DATA]
            elif 'exception' in result[rest_client.RESP_DATA]:
                raise RunnableException(result[rest_client.RESP_DATA]['exception']['message'], result[rest_client.RESP_STR])
            else:
                raise RunnableException('The status returned ' + str(result[rest_client.RESP_DATA]['status']), result[rest_client.RESP_STR])
        else:
            raise RunnableException(result[rest_client.RESP_REASON], result[rest_client.RESP_STR])
        return result_to_return