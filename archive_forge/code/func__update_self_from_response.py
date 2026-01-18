from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def _update_self_from_response(self, response, url, data):
    self._last_response_raw = response
    if isinstance(response['result'], list):
        result = response['result'][0]
    else:
        result = response['result']
    if 'status' in result:
        self._last_response_code = result['status']['code']
        self._last_response_msg = result['status']['message']
        self._last_url = url
        self._last_data_payload = data