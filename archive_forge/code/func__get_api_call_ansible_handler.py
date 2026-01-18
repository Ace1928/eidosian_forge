from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.urls import open_url
def _get_api_call_ansible_handler(self, method='get', resource_url='', stat_codes=None, params=None, payload_data=None):
    """
        Perform the HTTPS request by using ansible get/delete method
        """
    stat_codes = [200] if stat_codes is None else stat_codes
    request_url = str(self.base_url) + str(resource_url)
    response = None
    headers = {'Content-Type': 'application/json'}
    if not request_url:
        self.module.exit_json(msg='When sending Rest api call , the resource URL is empty, please check.')
    if payload_data and (not isinstance(payload_data, str)):
        payload_data = json.dumps(payload_data)
    response_raw = open_url(str(request_url), method=method, timeout=20, headers=headers, url_username=self.auth_user, url_password=self.auth_pass, validate_certs=False, force_basic_auth=True, data=payload_data)
    response = response_raw.read()
    payload = ''
    if response_raw.code not in stat_codes:
        self.module.exit_json(changed=False, meta=' openurl response_raw.code show error and error code is %r' % response_raw.code)
    elif isinstance(response, str) and len(response) > 0:
        payload = response
    elif method.lower() == 'delete' and response_raw.code == 204:
        payload = 'Delete is done.'
    if isinstance(payload, dict) and 'text' in payload:
        self.module.exit_json(changed=False, meta='when calling rest api, returned data is not json ')
        raise Exception(payload['text'])
    return payload