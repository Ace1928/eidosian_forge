from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
import logging
def add_account(module):
    logging.debug('Adding Account')
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    result = {}
    HTTPMethod = 'POST'
    end_point = '/PasswordVault/api/Accounts'
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    payload = {'safeName': module.params['safe']}
    for parameter_name in list(module.params.keys()):
        if parameter_name not in ansible_specific_parameters and module.params[parameter_name] is not None:
            cyberark_property_name = referenced_value(parameter_name, cyberark_reference_fieldnames, default=parameter_name)
            if isinstance(module.params[parameter_name], dict):
                payload[cyberark_property_name] = {}
                for dict_key in list(module.params[parameter_name].keys()):
                    cyberark_child_property_name = referenced_value(dict_key, cyberark_reference_fieldnames, default=dict_key)
                    logging.debug('parameter_name =%s.%s cyberark_property_name=%s cyberark_child_property_name=%s', parameter_name, dict_key, cyberark_property_name, cyberark_child_property_name)
                    if parameter_name + '.' + dict_key not in ansible_specific_parameters and module.params[parameter_name][dict_key] is not None:
                        payload[cyberark_property_name][cyberark_child_property_name] = deep_get(module.params[parameter_name], dict_key, _empty, False)
            elif parameter_name not in cyberark_reference_fieldnames:
                module_parm_value = deep_get(module.params, parameter_name, _empty, False)
                if module_parm_value is not None and module_parm_value != removal_value:
                    payload[parameter_name] = module_parm_value
            else:
                module_parm_value = deep_get(module.params, parameter_name, _empty, True)
                if module_parm_value is not None and module_parm_value != removal_value:
                    payload[cyberark_reference_fieldnames[parameter_name]] = module_parm_value
            logging.debug('parameter_name =%s', parameter_name)
    logging.debug('Add Account Payload => %s', json.dumps(payload))
    try:
        if module.check_mode:
            logging.debug('Proceeding with Add Account (CHECK_MODE)')
            return (True, {'result': None}, -1)
        else:
            logging.debug('Proceeding with Add Account')
            response = open_url(api_base_url + end_point, method=HTTPMethod, headers=headers, data=json.dumps(payload), validate_certs=validate_certs)
            result = {'result': json.loads(response.read())}
            return (True, result, response.getcode())
    except (HTTPError, HTTPException) as http_exception:
        if isinstance(http_exception, HTTPError):
            res = json.load(http_exception)
        else:
            res = to_text(http_exception)
        module.fail_json(msg='Error while performing add_account.Please validate parameters provided.\n*** end_point=%s%s\n ==> %s' % (api_base_url, end_point, res), payload=payload, headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing add_account.\n*** end_point=%s%s\n%s' % (api_base_url, end_point, to_text(unknown_exception)), payload=payload, headers=headers, status_code=-1)