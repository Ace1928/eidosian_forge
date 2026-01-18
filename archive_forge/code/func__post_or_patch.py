from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
def _post_or_patch(self, api_call, method, data, filter_none=True):
    if self._api_url not in api_call:
        api_endpoint = self._api_url + api_call
    else:
        api_endpoint = api_call
    headers = self._auth_header.copy()
    if data is not None:
        for k, v in deepcopy(data).items():
            if filter_none and v is None:
                del data[k]
        data = self._module.jsonify(data)
        headers['Content-type'] = 'application/json'
    resp, info = fetch_url(self._module, api_endpoint, headers=headers, method=method, data=data, timeout=self._module.params['api_timeout'])
    if info['status'] in (200, 201):
        return self._module.from_json(to_text(resp.read(), errors='surrogate_or_strict'))
    elif info['status'] == 204:
        return None
    else:
        self._module.fail_json(msg='Failure while calling the cloudscale.ch API with %s for "%s".' % (method, api_endpoint), fetch_url_info=info)