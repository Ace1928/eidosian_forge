from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
class OMEAlertPolicyInfo:

    def __init__(self) -> None:
        self.module = get_module_parameters()

    def get_all_alert_policy_info(self, rest_obj) -> dict:
        resp = rest_obj.invoke_request('GET', ALERT_POLICY_URI)
        value = resp.json_data['value']
        output_all = {'msg': MODULE_SUCCESS_MESSAGE_ALL, 'value': remove_key(value)}
        return output_all

    def get_alert_policy_info(self, rest_obj) -> dict:
        policy_name = self.module.params.get('policy_name')
        if policy_name is not None:
            output_not_found_or_empty = {'msg': POLICY_NAME_NOT_FOUND_OR_EMPTY.format(policy_name), 'value': []}
            if policy_name == '':
                return output_not_found_or_empty
            policies = self.get_all_alert_policy_info(rest_obj)
            for each_element in policies['value']:
                if each_element['Name'] == policy_name:
                    output_specific = {'msg': MODULE_SUCCESS_MESSAGE_SPECIFIC.format(policy_name), 'value': [each_element]}
                    return output_specific
            return output_not_found_or_empty
        return self.get_all_alert_policy_info(rest_obj)

    def perform_module_operation(self) -> None:
        try:
            with RestOME(self.module.params, req_session=True) as rest_obj:
                result = self.get_alert_policy_info(rest_obj)
                self.module.exit_json(msg=result['msg'], policies=result['value'])
        except HTTPError as err:
            self.module.exit_json(msg=str(err), error_info=json.load(err), failed=True)
        except URLError as err:
            self.module.exit_json(msg=str(err), unreachable=True)
        except (SSLValidationError, ConnectionError, TypeError, ValueError, OSError) as err:
            self.module.exit_json(msg=str(err), failed=True)